#ifndef __Eigen_HPP__
#define __Eigen_HPP__

#include <chrono>
#include <cstring>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

#include "memory_transfers.hpp"
#include "streaming_eigen.hpp"
#include "tuple.hpp"

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class EigenDDRToLocalMem;
class Eigen;
class EigenLocalMemToDDRQ;
class EigenLocalMemToDDRR;
class APipe;
class QPipe;
class RPipe;

/*
  Implementation of an Eigen values/vectors solver using multiple streaming
  kernels Can be configured by datatype, matrix size and works with real square
  matrices.
*/
template <unsigned k_size,       // Number of rows in the input matrix
          unsigned raw_latency,  // RAW latency for triangular loop optimization
          typename T             // The datatype for the computation
          >
void EigenImpl(
    std::vector<T> &a_matrix,             // Input matrix to decompose
    std::vector<T> &q_matrix,             // Output matrix Q
    std::vector<T> &eigen_values_matrix,  // Output Eigen values
    sycl::queue &q,                       // Device queue
    int matrix_count,                     // Number of matrices to decompose
    int repetitions  // Number of repetitions, for performance evaluation
) {
  constexpr int kMatrixSize = k_size * k_size;
  constexpr int kEValuesSize = k_size;
  constexpr int kNumElementsPerDDRBurst = 8;

  using PipeType = fpga_tools::NTuple<T, kNumElementsPerDDRBurst>;

  // Pipes to communicate the A, Q and R matrices between kernels
  using AMatrixPipe = sycl::ext::intel::pipe<APipe, PipeType, 3>;
  using QMatrixPipe = sycl::ext::intel::pipe<QPipe, PipeType, 3>;
  using EValuesPipe =
      sycl::ext::intel::pipe<RPipe, T, kNumElementsPerDDRBurst * 4>;

  // Allocate FPGA DDR memory.
#if defined(IS_BSP)
  T *a_device = sycl::malloc_device<T>(kMatrixSize * matrix_count, q);
  T *q_device = sycl::malloc_device<T>(kMatrixSize * matrix_count, q);
  T *eigen_values_device =
      sycl::malloc_device<T>(kEValuesSize * matrix_count, q);
#else
  // malloc_device are not supported when targeTing an FPGA part/family
  T *a_device = sycl::malloc_shared<T>(kMatrixSize * matrix_count, q);
  T *q_device = sycl::malloc_shared<T>(kMatrixSize * matrix_count, q);
  T *eigen_values_device =
      sycl::malloc_shared<T>(kEValuesSize * matrix_count, q);
#endif

  if (a_device == nullptr) {
    std::cout << "Failed to allocated memory for the A matrix" << std::endl;
    std::terminate();
  }

  std::cout << "kMatrixSize " << kMatrixSize << std::endl;
  std::cout << "matrix_count " << matrix_count << std::endl;
  std::cout << "sizeof(T) " << int(sizeof(T)) << std::endl;

  q.memcpy(a_device, a_matrix.data(), kMatrixSize * matrix_count * sizeof(T))
      .wait();

  auto ddr_write_event = q.submit([&](sycl::handler &h) {
    h.single_task<EigenDDRToLocalMem>([=]() [[intel::kernel_args_restrict]] {
      MatrixReadFromDDRToPipe<T, k_size, k_size, kNumElementsPerDDRBurst,
                              AMatrixPipe>(a_device, matrix_count, repetitions);
    });
  });

  // Read the A matrix from the AMatrixPipe pipe and compute the QR
  // decomposition. Write the Q and R output matrices to the QMatrixPipe
  // and EValuesPipe pipes.
  q.single_task<Eigen>(
      fpga_linalg::StreamingEigen<T, k_size, kNumElementsPerDDRBurst,
                                  AMatrixPipe, QMatrixPipe, EValuesPipe>());

  auto q_event =
      q.single_task<EigenLocalMemToDDRQ>([=]() [[intel::kernel_args_restrict]] {
        // Read the Q matrix from the QMatrixPipe pipe and copy it to the
        // FPGA DDR
        MatrixReadPipeToDDR<T, k_size, k_size, kNumElementsPerDDRBurst,
                            QMatrixPipe>(q_device, matrix_count, repetitions);
      });

  auto r_event = q.single_task<EigenLocalMemToDDRR>([=
  ]() [[intel::kernel_args_restrict]] {
  // Read the R matrix from the EValuesPipe pipe and copy it to the
  // FPGA DDR

#if defined(IS_BSP)
    // When targeting a BSP, we instruct the compiler that this pointer
    // lives on the device.
    // Knowing this, the compiler won't generate hardware to
    // potentially get data from the host.
    sycl::device_ptr<T> vector_ptr_located(eigen_values_device);
#else
    // Device pointers are not supported when targeting an FPGA 
    // family/part
    T* vector_ptr_located(eigen_values_device);
#endif

    // Repeat matrix_count complete R matrix pipe reads
    // for as many repetitions as needed
    for (int repetition_index = 0; repetition_index < repetitions;
         repetition_index++) {
      [[intel::loop_coalesce(2)]]  // NO-FORMAT: ATribute
      for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++) {
        for (int r_idx = 0; r_idx < kEValuesSize; r_idx++) {
          vector_ptr_located[matrix_index * kEValuesSize + r_idx] =
              EValuesPipe::read();
        }  // end of r_idx
      }    // end of repetition_index
    }      // end of li
  });

  q_event.wait();
  r_event.wait();

  // Compute the total time the execution lasted
  auto start_time = ddr_write_event.template get_profiling_info<
      sycl::info::event_profiling::command_start>();
  auto end_time = q_event.template get_profiling_info<
      sycl::info::event_profiling::command_end>();
  double diff = (end_time - start_time) / 1.0e9;
  q.throw_asynchronous();

  std::cout << "   Total duration:   " << diff << " s" << std::endl;
  std::cout << "Throughput: " << repetitions * matrix_count / diff * 1e-3
            << "k matrices/s" << std::endl;

  // Copy the Q and R matrices result from the FPGA DDR to the host memory
  q.memcpy(q_matrix.data(), q_device, kMatrixSize * matrix_count * sizeof(T))
      .wait();
  q.memcpy(eigen_values_matrix.data(), eigen_values_device,
           kEValuesSize * matrix_count * sizeof(T))
      .wait();

  // Clean allocated FPGA memory
  free(a_device, q);
  free(q_device, q);
  free(eigen_values_device, q);
}

#endif /* __Eigen_HPP__ */
