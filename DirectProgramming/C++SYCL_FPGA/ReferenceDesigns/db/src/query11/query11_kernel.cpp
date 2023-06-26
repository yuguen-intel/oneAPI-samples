#include <stdio.h>

#include <type_traits>

#include "query11_kernel.hpp"
#include "pipe_types.hpp"
#include "../db_utils/CachedMemory.hpp"
#include "../db_utils/MapJoin.hpp"
#include "../db_utils/Misc.hpp"
#include "../db_utils/Tuple.hpp"
#include "../db_utils/Unroller.hpp"
#include "../db_utils/fifo_sort.hpp"

#include "onchip_memory_with_cache.hpp" // DirectProgramming/C++SYCL_FPGA/include

using namespace std::chrono;

// kernel class names
class ProducePartSupplier;
class JoinPartSupplierParts;
class Compute;
class FifoSort;
class ConsumeSort;

///////////////////////////////////////////////////////////////////////////////
// sort configuration
using SortType = OutputData;
constexpr int kNumSortStages = CeilLog2(kPartTableSize);
constexpr int kSortSize = Pow2(kNumSortStages);

static_assert(kPartTableSize <= kSortSize,
              "Must be able to sort all part keys");

// comparator for the sorter to sort in descending order
struct GreaterThan {
  inline bool operator()(const SortType& a, const SortType& b) {
    return a.partvalue > b.partvalue;
  }
};

// input and output pipes for the sorter
using SortInPipe = pipe<class SortInputPipe, SortType>;
using SortInPipeSpy3 = pipe<class SortInputPipeSpy3, SortType>;
using SortOutPipe = pipe<class SortOutputPipe, SortType>;
using SortOutPipeSpy4 = pipe<class SortOutputPipeSpy4, SortType>;
///////////////////////////////////////////////////////////////////////////////

bool SubmitQuery11(queue& q, Database& dbinfo, std::string& nation,
                    std::vector<DBIdentifier>& partkeys,
                    std::vector<DBDecimal>& values,
                    double& kernel_latency, double& total_latency) {
  // find the nationkey based on the nation name
  // assert(dbinfo.n.name_key_map.find(nation) != dbinfo.n.name_key_map.end());
  unsigned char nationkey = dbinfo.n.name_key_map[nation];

  // ensure correctly sized output buffers
  partkeys.resize(kPartTableSize);
  values.resize(kPartTableSize);

  // create space for the input buffers
  // SUPPLIER
  buffer s_nationkey_buf(dbinfo.s.nationkey);
  
  // PARTSUPPLIER
  buffer ps_partkey_buf(dbinfo.ps.partkey);
  buffer ps_suppkey_buf(dbinfo.ps.suppkey);
  buffer ps_availqty_buf(dbinfo.ps.availqty);
  buffer ps_supplycost_buf(dbinfo.ps.supplycost);

  // setup the output buffers
  buffer partkeys_buf(partkeys);
  buffer values_buf(values);


  constexpr int kSpy1BufferSize = 1000000;
  PartSupplierRowPipeData * spy_1_data = sycl::malloc_device<PartSupplierRowPipeData>(kSpy1BufferSize, q);
  int * spy_1_count = sycl::malloc_device<int>(1, q);

  constexpr int kSpy2BufferSize = 1000000;
  SupplierPartSupplierJoinedPipeData * spy_2_data = sycl::malloc_device<SupplierPartSupplierJoinedPipeData>(kSpy2BufferSize, q);
  int * spy_2_count = sycl::malloc_device<int>(1, q);

  constexpr int kSpy3BufferSize = 1000000;
  SortType * spy_3_data = sycl::malloc_device<SortType>(kSpy3BufferSize, q);
  int * spy_3_count = sycl::malloc_device<int>(1, q);

  constexpr int kSpy4BufferSize = 1000000;
  SortType * spy_4_data = sycl::malloc_device<SortType>(kSpy4BufferSize, q);
  int * spy_4_count = sycl::malloc_device<int>(1, q);

  if(spy_1_data == nullptr ||
     spy_1_count == nullptr ||
     spy_2_data == nullptr ||
     spy_2_count == nullptr ||
     spy_3_data == nullptr ||
     spy_3_count == nullptr ||
     spy_4_data == nullptr ||
     spy_4_count == nullptr
    )
  {
    std::cerr << "Device memory allocation failure" << std::endl;
    std::terminate();
  }

  // number of producing iterations depends on the number of elements per cycle
  const size_t ps_rows = dbinfo.ps.rows;
  const size_t ps_iters = (ps_rows + kJoinWinSize - 1) / kJoinWinSize;

  // start timer
  high_resolution_clock::time_point host_start = high_resolution_clock::now();

  ///////////////////////////////////////////////////////////////////////////
  //// ProducePartSupplier Kernel
  auto produce_ps_event = q.submit([&](handler& h) {
    // PARTSUPPLIER table accessors
    accessor ps_partkey_accessor(ps_partkey_buf, h, read_only);
    accessor ps_suppkey_accessor(ps_suppkey_buf, h, read_only);
    accessor ps_availqty_accessor(ps_availqty_buf, h, read_only);
    accessor ps_supplycost_accessor(ps_supplycost_buf, h, read_only);

    // kernel to produce the PARTSUPPLIER table
    h.single_task<ProducePartSupplier>([=]() [[intel::kernel_args_restrict]] {
      [[intel::initiation_interval(1)]]
      for (size_t i = 0; i < ps_iters; i++) {
        // bulk read of data from global memory
        NTuple<kJoinWinSize, PartSupplierRow> data;

        UnrolledLoop<0, kJoinWinSize>([&](auto j) {
          size_t idx = i * kJoinWinSize + j;
          bool in_range = idx < ps_rows;

          DBIdentifier partkey = ps_partkey_accessor[idx];
          DBIdentifier suppkey = ps_suppkey_accessor[idx];
          int availqty = ps_availqty_accessor[idx];
          DBDecimal supplycost = ps_supplycost_accessor[idx];

          data.get<j>() =
              PartSupplierRow(in_range, partkey, suppkey, availqty, supplycost);
        });

        // write to pipe
        ProducePartSupplierPipe::write(
            PartSupplierRowPipeData(false, true, data));
      }

      // tell the downstream kernel we are done producing data
      ProducePartSupplierPipe::write(PartSupplierRowPipeData(true, false));
    });
  });
  ///////////////////////////////////////////////////////////////////////////

  // Spy kernel 
  q.submit([&](handler& h) {
    h.single_task<class Spy1>([=]() [[intel::kernel_args_restrict]] {
      int idx = 0;
      while(1){
        auto read = ProducePartSupplierPipe::read();
        spy_1_data[idx] = read;
        idx++;
        spy_1_count[0] = idx;
        ProducePartSupplierPipeSpy1::write(read);
      }
    });
  });

  ///////////////////////////////////////////////////////////////////////////
  //// JoinPartSupplierParts Kernel
  auto join_event = q.submit([&](handler& h) {
    // SUPPLIER table accessors
    size_t s_rows = dbinfo.s.rows;
    accessor s_nationkey_accessor(s_nationkey_buf, h, read_only);

    h.single_task<JoinPartSupplierParts>([=]() [[intel::kernel_args_restrict]] {
      // initialize the array map
      // +1 is to account for fact that SUPPKEY is [1,kSF*10000]
      unsigned char nation_key_map_data[kSupplierTableSize + 1];
      bool nation_key_map_valid[kSupplierTableSize + 1];
      for (int i = 0; i < kSupplierTableSize + 1; i++) {
        nation_key_map_valid[i] = false;
      }

      // populate MapJoiner map
      // why a map? keys may not be sequential
      [[intel::initiation_interval(1)]]
      for (size_t i = 0; i < s_rows; i++) {
        // NOTE: based on TPCH docs, SUPPKEY is guaranteed to be unique
        // in the range [1:kSF*10000]
        DBIdentifier s_suppkey = i + 1;
        unsigned char s_nationkey = s_nationkey_accessor[i];
        
        nation_key_map_data[s_suppkey] = s_nationkey;
        nation_key_map_valid[s_suppkey] = true;
      }

      // MAPJOIN PARTSUPPLIER and SUPPLIER tables by suppkey
      MapJoin<unsigned char, ProducePartSupplierPipeSpy1, PartSupplierRow,
              kJoinWinSize, PartSupplierPartsPipe,
              SupplierPartSupplierJoined>(nation_key_map_data,
                                          nation_key_map_valid);
      
      // tell downstream we are done
      PartSupplierPartsPipe::write(
          SupplierPartSupplierJoinedPipeData(true,false));
    });
  });
  ///////////////////////////////////////////////////////////////////////////


  // Spy kernel 
  q.submit([&](handler& h) {
    h.single_task<class Spy2>([=]() [[intel::kernel_args_restrict]] {
      int idx = 0;
      while(1){
        auto read = PartSupplierPartsPipe::read();
        spy_2_data[idx] = read;
        idx++;
        spy_2_count[0] = idx;
        PartSupplierPartsPipeSpy2::write(read);
      }
    });
  });


  ///////////////////////////////////////////////////////////////////////////
  //// Compute Kernel
  auto compute_event = q.single_task<Compute>([=] {
    constexpr int kAccumCacheSize = 15;
    fpga_tools::OnchipMemoryWithCache<DBDecimal, kPartTableSize, 
                                      kAccumCacheSize> partkey_values;

    // initialize accumulator
    partkey_values.init(0);

    bool done = false;

    [[intel::initiation_interval(1)]]
    while (!done) {
      bool valid_pipe_read;
      SupplierPartSupplierJoinedPipeData pipe_data = 
          PartSupplierPartsPipeSpy2::read(valid_pipe_read);

      done = pipe_data.done && valid_pipe_read;

      if (valid_pipe_read && !done) {
        UnrolledLoop<0, kJoinWinSize>([&](auto j) {
          SupplierPartSupplierJoined data = pipe_data.data.template get<j>();

          if (data.valid && data.nationkey == nationkey) {
            // partkeys start at 1
            DBIdentifier index = data.partkey - 1;
            DBDecimal val = data.supplycost * (DBDecimal)(data.availqty);
            auto curr_val = partkey_values.read(index);
            partkey_values.write(index, curr_val + val);
          }
        });
      }
    }

    // sort the {partkey, partvalue} pairs based on partvalue.
    // we will send in kSortSize - kPartTableSize dummy values with a
    // minimum value so that they are last (sorting from highest to lowest)
    [[intel::initiation_interval(1)]]
    for (size_t i = 0; i < kSortSize; i++) {
      size_t key = (i < kPartTableSize) ? (i + 1) : 0;
      auto val = (i < kPartTableSize) ? partkey_values.read(i)
                                      : std::numeric_limits<DBDecimal>::min();
      SortInPipe::write(OutputData(key, val));
    }
  });
  ///////////////////////////////////////////////////////////////////////////

  // Spy kernel 
  q.submit([&](handler& h) {
    h.single_task<class Spy3>([=]() [[intel::kernel_args_restrict]] {
      int idx = 0;
      while(1){
        auto read = SortInPipe::read();
        spy_3_data[idx] = read;
        idx++;
        spy_3_count[0] = idx;
        SortInPipeSpy3::write(read);
      }
    });
  });



  ///////////////////////////////////////////////////////////////////////////
  //// ConsumeSort kernel
  auto consume_sort_event = q.submit([&](handler& h) {
    // output buffer accessors
    accessor partkeys_accessor(partkeys_buf, h, write_only, no_init);
    accessor values_accessor(values_buf, h, write_only, no_init);

    h.single_task<ConsumeSort>([=]() [[intel::kernel_args_restrict]] {
      int i = 0;
      bool i_in_range = 0 < kSortSize;
      bool i_next_in_range = 1 < kSortSize;
      bool i_in_parttable_range = 0 < kPartTableSize;
      bool i_next_in_parttable_range = 1 < kPartTableSize;

      // grab all kSortSize elements from the sorter
      [[intel::initiation_interval(1)]]
      while (i_in_range) {
        bool pipe_read_valid;
        OutputData D = SortOutPipeSpy4::read(pipe_read_valid);

        if (pipe_read_valid) {
          if (i_in_parttable_range) {
            partkeys_accessor[i] = D.partkey;
            values_accessor[i] = D.partvalue;
          }

          i_in_range = i_next_in_range;
          i_next_in_range = i < kSortSize - 2;
          i_in_parttable_range = i_next_in_parttable_range;
          i_next_in_parttable_range = i < kPartTableSize - 2;
          i++;
        }
      }
    });
  });
  ///////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////
  //// FifoSort Kernel
  auto sort_event = q.single_task<FifoSort>([=] {
    ihc::sort<SortType, kSortSize, SortInPipeSpy3, SortOutPipe>(GreaterThan());
  });
  ///////////////////////////////////////////////////////////////////////////


  // Spy kernel 
  q.submit([&](handler& h) {
    h.single_task<class Spy4>([=]() [[intel::kernel_args_restrict]] {
      int idx = 0;
      while(1){
        auto read = SortOutPipe::read();
        spy_4_data[idx] = read;
        idx++;
        spy_4_count[0] = idx;
        SortOutPipeSpy4::write(read);
      }
    });
  });

  // wait for kernels to finish
  produce_ps_event.wait();
  join_event.wait();
  compute_event.wait();
  sort_event.wait();
  consume_sort_event.wait();

  high_resolution_clock::time_point host_end = high_resolution_clock::now();
  duration<double, std::milli> diff = host_end - host_start;



  PartSupplierRowPipeData* spy_1_data_host = (PartSupplierRowPipeData*) malloc(kSpy1BufferSize * sizeof(PartSupplierRowPipeData));
  if (spy_1_data_host == nullptr){
    std::cerr << "Failed to allocate host data memory" << std::endl;
    std::terminate();
  }
  int spy_1_data_count_host;
  
  std::cout << "Copying from device to host" << std::endl;

  q.memcpy(spy_1_data_host, spy_1_data, kSpy1BufferSize * sizeof(PartSupplierRowPipeData)).wait();
  q.memcpy(&spy_1_data_count_host, spy_1_count, sizeof(int)).wait();

  std::cout << "Copy over" << std::endl;

  std::ofstream myfile_spy_1;
  myfile_spy_1.open ("spy_1.txt");

  for (int i =0; i<spy_1_data_count_host; i++){
    auto current_data = spy_1_data_host[i];
    PartSupplierRow current_row = current_data.data.template get<0>();

    std::string valid_str = current_row.valid ? "true" : "false";
    unsigned int partkey = current_row.partkey;
    unsigned int suppkey = current_row.suppkey;
    int availqty = current_row.availqty;
    long long supplycost = current_row.supplycost;
    myfile_spy_1 << i << "/" << (spy_1_data_count_host-1) << std::endl;
    myfile_spy_1 << valid_str << " "
              << partkey << " "
              << suppkey << " "
              << availqty << " "
              << supplycost << " "
              << std::endl;

  }
  myfile_spy_1.close();
  std::cout << "Freeing memory" << std::endl;
  free(spy_1_data_host);

  std::cout << "wrote to spy_1.txt" << std::endl;

  SupplierPartSupplierJoinedPipeData* spy_2_data_host = (SupplierPartSupplierJoinedPipeData*) malloc(kSpy1BufferSize * sizeof(SupplierPartSupplierJoinedPipeData));
  if (spy_2_data_host == nullptr){
    std::cerr << "Failed to allocate host data memory" << std::endl;
    std::terminate();
  }
  int spy_2_data_count_host;

  std::cout << "Copying from device to host" << std::endl;

  q.memcpy(spy_2_data_host, spy_2_data, kSpy1BufferSize * sizeof(SupplierPartSupplierJoinedPipeData)).wait();
  q.memcpy(&spy_2_data_count_host, spy_2_count, sizeof(int)).wait();
  std::cout << "Copy over" << std::endl;


  std::ofstream myfile_spy_2;
  myfile_spy_2.open ("spy_2.txt");

  for (int i =0; i<spy_2_data_count_host; i++){
    auto current_data = spy_2_data_host[i];
    SupplierPartSupplierJoined current = current_data.data.template get<0>();

    std::string valid_str = current.valid ? "true" : "false";
    unsigned int partkey = current.partkey;
    int availqty = current.availqty;
    long long supplycost = current.supplycost;
    std::string nationkey = std::to_string(current.nationkey);
    myfile_spy_2 << i << "/" << (spy_2_data_count_host-1) << std::endl;
    myfile_spy_2 << valid_str << " "
              << partkey << " "
              << availqty << " "
              << supplycost << " "
              << nationkey << " "
              << std::endl;

  }
  myfile_spy_2.close();
  std::cout << "Freeing memory" << std::endl;

  free(spy_2_data_host);

  std::cout << "wrote to spy_2.txt" << std::endl;


  SortType* spy_3_data_host = (SortType*) malloc(kSpy1BufferSize * sizeof(SortType));
  if (spy_3_data_host == nullptr){
    std::cerr << "Failed to allocate host data memory" << std::endl;
    std::terminate();
  }
  int spy_3_data_count_host;

  std::cout << "Copying from device to host" << std::endl;

  q.memcpy(spy_3_data_host, spy_3_data, kSpy1BufferSize * sizeof(SortType)).wait();
  q.memcpy(&spy_3_data_count_host, spy_3_count, sizeof(int)).wait();
  std::cout << "Copy over" << std::endl;


  std::ofstream myfile_spy_3;
  myfile_spy_3.open ("spy_3.txt");

  for (int i =0; i<spy_3_data_count_host; i++){
    auto current_data = spy_3_data_host[i];

    unsigned int partkey = current_data.partkey;
    long long partvalue = current_data.partvalue;
    myfile_spy_3 << i << "/" << (spy_3_data_count_host-1) << std::endl;
    myfile_spy_3 << partkey << " "
              << partvalue << " "
              << std::endl;

  }
  myfile_spy_3.close();
  std::cout << "Freeing memory" << std::endl;
  free(spy_3_data_host);

  std::cout << "wrote to spy_3.txt" << std::endl;


  SortType* spy_4_data_host = (SortType*) malloc(kSpy1BufferSize*sizeof(SortType));
  if (spy_4_data_host == nullptr){
    std::cerr << "Failed to allocate host data memory" << std::endl;
    std::terminate();
  }
  int spy_4_data_count_host;

  std::cout << "Copying from device to host" << std::endl;

  q.memcpy(spy_4_data_host, spy_4_data, kSpy1BufferSize * sizeof(SortType)).wait();
  q.memcpy(&spy_4_data_count_host, spy_4_count, sizeof(int)).wait();

  std::cout << "Copy over" << std::endl;

  std::ofstream myfile_spy_4;
  myfile_spy_4.open ("spy_4.txt");

  for (int i =0; i<spy_4_data_count_host; i++){
    auto current_data = spy_4_data_host[i];

    unsigned int partkey = current_data.partkey;
    long long partvalue = current_data.partvalue;
    myfile_spy_4 << i << "/" << (spy_4_data_count_host-1) << std::endl;
    myfile_spy_4 << partkey << " "
              << partvalue << " "
              << std::endl;

  }
  myfile_spy_4.close();
  std::cout << "Freeing memory" << std::endl;
  free(spy_4_data_host);

  std::cout << "wrote to spy_4.txt" << std::endl;

  // free(spy_1_data, q);
  // free(spy_1_count, q);
  // free(spy_2_data, q);
  // free(spy_2_count, q);
  // free(spy_3_data, q);
  // free(spy_3_count, q);
  // free(spy_4_data, q);
  // free(spy_4_count, q);


  // gather profiling info
  auto start_time =
      consume_sort_event
          .get_profiling_info<info::event_profiling::command_start>();
  auto end_time = consume_sort_event
          .get_profiling_info<info::event_profiling::command_end>();


  // calculating the kernel execution time in ms
  auto kernel_execution_time = (end_time - start_time) * 1e-6;
  std::cout << "Kernel time: " << kernel_execution_time << std::endl;

  kernel_latency = kernel_execution_time;
  total_latency = diff.count();

  return true;
}
