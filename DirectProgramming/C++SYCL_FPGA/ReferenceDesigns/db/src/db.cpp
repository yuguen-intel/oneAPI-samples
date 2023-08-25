

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

#define PRINTF(format, ...)                                          \
  {                                                                  \
    static const CL_CONSTANT char _format[] = format;                \
    sycl::ext::oneapi::experimental::printf(_format, ##__VA_ARGS__); \
  }

#include "exception_handler.hpp"

using namespace sycl;

// database types
using DBIdentifier = unsigned int;
using DBDecimal = long long;

constexpr int kPartTableSize = 200000;

#include "db_utils/fifo_sort.hpp"

class OutputData {
 public:
  OutputData() {}
  // OutputData() : partkey(0), partvalue(0) {}
  OutputData(DBIdentifier v_partkey, DBDecimal v_partvalue)
      : partkey(v_partkey), partvalue(v_partvalue) {}

  bool operator<(const OutputData& t) const { return partvalue < t.partvalue; }
  bool operator>(const OutputData& t) const { return partvalue > t.partvalue; }
  bool operator==(const OutputData& t) const {
    return partvalue == t.partvalue;
  }
  bool operator!=(const OutputData& t) const {
    return partvalue != t.partvalue;
  }

  DBIdentifier partkey;
  DBDecimal partvalue;
};

class FifoSort;

///////////////////////////////////////////////////////////////////////////////
// sort configuration
constexpr int kNumSortStages = CeilLog2(kPartTableSize);
constexpr int kSortSize = Pow2(kNumSortStages);

// comparator for the sorter to sort in descending order
struct GreaterThan {
  inline bool operator()(const OutputData& a, const OutputData& b) {
    return a.partvalue > b.partvalue;
  }
};

// input and output pipes for the sorter
using SortInPipeSpy3 = pipe<class SortInputPipeSpy3, OutputData>;
using SortOutPipe = pipe<class SortOutputPipe, OutputData>;
///////////////////////////////////////////////////////////////////////////////

//
// main
//
int main(int argc, char* argv[]) {
  try {
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    // create the device queue
    queue q(selector, fpga_tools::exception_handler);

    device device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<info::device::name>().c_str() << std::endl;

    /*
        Read golden inputs
      */
    std::cout << "Golden data spy_3" << std::endl;
    constexpr int kSpy3BufferSize = 262145;

    OutputData* spy_3_data_host = (OutputData*)malloc(262145 * sizeof(OutputData));
    if (spy_3_data_host == nullptr) {
      std::cerr << "Failed to allocate host data memory" << std::endl;
      std::terminate();
    }

    int index_3 = 0;
    std::string line_3;
    std::ifstream myfile_3("spy_3.txt");
    if (myfile_3.is_open()) {
      while (getline(myfile_3, line_3)) {
        if (index_3 % 2 == 1) {
          OutputData current_data{};

          std::istringstream is(line_3);

          unsigned int partkey;
          is >> partkey;

          long long partvalue;
          is >> partvalue;

          // std::cout << index_3/2 << "/262144" << std::endl;
          // std::cout << partkey << " " << partvalue << std::endl;

          current_data.partkey = partkey;
          current_data.partvalue = partvalue;

          spy_3_data_host[index_3 / 2] = current_data;
        }

        index_3++;
      }

      myfile_3.close();
    } else {
      std::cerr << "Failed to spy3.txt" << std::endl;
      std::terminate();
    }

    OutputData* spy_3_data = sycl::malloc_device<OutputData>(kSpy3BufferSize, q);

    if (spy_3_data == nullptr) {
      std::cerr << "Device memory allocation failure" << std::endl;
      std::terminate();
    }

    std::cout << "Copying from host to device" << std::endl;

    q.memcpy(spy_3_data, spy_3_data_host, kSpy3BufferSize * sizeof(OutputData))
        .wait();
    std::cout << "Copy over" << std::endl;

    /*
      Read golden outputs
    */

    std::cout << "Golden data spy_4" << std::endl;

    constexpr int kSpy4BufferSize = 262145;

    OutputData* spy_4_golden_data_host =
        (OutputData*)malloc(262145 * sizeof(OutputData));
    if (spy_4_golden_data_host == nullptr) {
      std::cerr << "Failed to allocate host data memory" << std::endl;
      std::terminate();
    }

    int index_4 = 0;
    std::string line_4;
    std::ifstream myfile_4("spy_4.txt");
    if (myfile_4.is_open()) {
      while (getline(myfile_4, line_4)) {
        if (index_4 % 2 == 1) {
          OutputData current_data{};

          std::istringstream is(line_4);

          unsigned int partkey;
          is >> partkey;

          long long partvalue;
          is >> partvalue;

          // std::cout << index_4/2 << "/262144" << std::endl;
          // std::cout << partkey << " " << partvalue << std::endl;

          current_data.partkey = partkey;
          current_data.partvalue = partvalue;

          spy_4_golden_data_host[index_4 / 2] = current_data;
        }

        index_4++;
      }

      myfile_4.close();
    } else {
      std::cerr << "Failed to spy3.txt" << std::endl;
      std::terminate();
    }

    OutputData* spy_4_golden_data =
        sycl::malloc_device<OutputData>(kSpy4BufferSize, q);

    if (spy_4_golden_data == nullptr) {
      std::cerr << "Device memory allocation failure" << std::endl;
      std::terminate();
    }

    std::cout << "Copying from host to device" << std::endl;

    q.memcpy(spy_4_golden_data, spy_4_golden_data_host,
             kSpy4BufferSize * sizeof(OutputData))
        .wait();
    std::cout << "Copy over" << std::endl;

    /*
      Allocate one int to check the result
    */
    int* spy_4_check = sycl::malloc_device<int>(1, q);

    if (spy_4_check == nullptr) {
      std::cerr << "Device memory allocation failure" << std::endl;
      std::terminate();
    }

    std::cout << "Kernel data" << std::endl;

    // Kernel that feeds the golden inputs
    q.submit([&](handler& h) {
      h.single_task<class Spy3>([=]() [[intel::kernel_args_restrict]] {
        for (int i = 0; i < 262144; i++) {
          auto data = spy_3_data[i];
          SortInPipeSpy3::write(data);
        }
      });
    });

    ///////////////////////////////////////////////////////////////////////////
    //// FifoSort Kernel
    auto sort_event = q.single_task<FifoSort>([=] {
      ihc::sort<OutputData, kSortSize, SortInPipeSpy3, SortOutPipe>(
          GreaterThan());
    });
    ///////////////////////////////////////////////////////////////////////////

    // Kernel that checks the outputs vs the golden outputs
    event spy4_event = q.submit([&](handler& h) {
      h.single_task<class Spy4>([=]() [[intel::kernel_args_restrict]] {
        int pass = 0;
        for (int i = 0; i < 262144; i++) {
          auto read = SortOutPipe::read();

          if (read.partvalue != spy_4_golden_data[i].partvalue ||
              read.partkey != spy_4_golden_data[i].partkey) {
            PRINTF("At index %d\n", i);
            PRINTF("Partkey; expecting %lld, got %lld\n",
                   spy_4_golden_data[i].partkey, read.partkey);
            PRINTF("Partvalue; expecting %lld, got %lld\n",
                   spy_4_golden_data[i].partvalue, read.partvalue);
            pass = 1;
          }
        }
        spy_4_check[0] = pass;
      });
    });

    // wait for kernels to finish
    spy4_event.wait();

    /*
      Copy the check int
    */
    std::array<int, 1> res;
    q.memcpy(res.data(), spy_4_check, sizeof(int)).wait();

    bool pass = res[0] == 0;
    std::cout << (pass ? "PASS" : "FAIL") << std::endl;

  } catch (exception const& e) {
    // Catches exceptions in the host code
    std::cout << "Caught a SYCL host exception:\n" << e.what() << "\n";
    std::terminate();
  }

  return 0;
}
