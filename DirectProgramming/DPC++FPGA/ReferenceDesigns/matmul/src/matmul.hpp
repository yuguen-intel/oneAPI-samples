#ifndef __MATMUL_HPP__
#define __MATMUL_HPP__

#include <CL/sycl.hpp>
#include <chrono>
#include <cstring>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <type_traits>
#include <vector>

#include "memory_transfers.hpp"
#include "streaming_matmul.hpp"
#include "tuple.hpp"

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class MATMULDDRToLocalMemA;
class MATMULDDRToLocalMemB;
class MATMUL;
class MATMULLocalMemToDDR;
class APipe;
class BPipe;
class MMPipe;

/*
  Implementation of the matrix multiplication using multiple streaming kernels
  Can be configured by datatype, matrix size and works with square or
  rectangular matrices, real and complex.
*/
template <unsigned columns,  // Number of columns in the input matrix
          unsigned rows,     // Number of rows in the input matrix
          bool is_complex,   // Selects between ac_complex<T> and T datatype
          typename T,        // The datatype for the computation
          typename TT = std::conditional_t<is_complex, ac_complex<T>, T>
          // TT will be ac_complex<T> or T depending on is_complex
          >
void MATMULImpl(
    std::vector<TT> &a_matrix,   // Input matrix A
    std::vector<TT> &b_matrix,   // Input matrix B
    std::vector<TT> &mm_matrix,  // Output matrix A*B
    sycl::queue &q,              // Device queue
    int matrix_count,            // Number of matrices to multiply
    int repetitions  // Number of repetitions, for performance evaluation
) {
  constexpr int kMatrixSize = columns * rows;
  constexpr int kNumElementsPerDDRBurst = is_complex ? 4 : 8;

  using PipeType = fpga_tools::NTuple<TT, kNumElementsPerDDRBurst>;

  // Pipes to communicate the A, Q and R matrices between kernels
  using AMatrixPipe = sycl::ext::intel::pipe<APipe, PipeType, 3>;
  using BMatrixPipe = sycl::ext::intel::pipe<BPipe, PipeType, 3>;
  using MMMatrixPipe = sycl::ext::intel::pipe<MMPipe, PipeType, 3>;

  // Allocate FPGA DDR memory.
  TT *a_device = sycl::malloc_device<TT>(kMatrixSize * matrix_count, q);
  TT *b_device = sycl::malloc_device<TT>(kMatrixSize * matrix_count, q);
  TT *mm_device = sycl::malloc_device<TT>(kMatrixSize * matrix_count, q);

  q.memcpy(a_device, a_matrix.data(), kMatrixSize * matrix_count * sizeof(TT))
      .wait();
  q.memcpy(b_device, b_matrix.data(), kMatrixSize * matrix_count * sizeof(TT))
      .wait();

  // Producer kernel for matrix A
  auto ddr_write_event = q.submit([&](sycl::handler &h) {
    h.single_task<MATMULDDRToLocalMemA>([=]() [[intel::kernel_args_restrict]] {
      MatrixReadFromDDRToPipe<TT, rows, columns, kNumElementsPerDDRBurst,
                              AMatrixPipe>(a_device, matrix_count, repetitions);
    });
  });

  // Producer kernel for matrix B
  q.submit([&](sycl::handler &h) {
    h.single_task<MATMULDDRToLocalMemB>([=]() [[intel::kernel_args_restrict]] {
      MatrixReadFromDDRToPipe<TT, rows, columns, kNumElementsPerDDRBurst,
                              BMatrixPipe>(b_device, matrix_count, repetitions);
    });
  });

  // Read the A matrix from the AMatrixPipe pipe
  // Read the B matrix from the BMatrixPipe pipe
  // Compute A*B and write the result to the MMMatrixPipe pipe
  q.single_task<MATMUL>(
      fpga_linalg::StreamingMatmul<T, is_complex, rows, columns,
                                   kNumElementsPerDDRBurst, AMatrixPipe,
                                   BMatrixPipe, MMMatrixPipe>());

  // Consumer kernel for the result matrix
  sycl::event q_event =
      q.single_task<MATMULLocalMemToDDR>([=]() [[intel::kernel_args_restrict]] {
         MatrixReadPipeToDDR<TT, rows, columns, kNumElementsPerDDRBurst,
                             MMMatrixPipe>(mm_device, matrix_count,
                                           repetitions);
       });

  q_event.wait();

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
  q.memcpy(mm_matrix.data(), mm_device, kMatrixSize * matrix_count * sizeof(TT))
      .wait();

  // Clean allocated FPGA memory
  free(a_device, q);
  free(b_device, q);
  free(mm_device, q);
}

#endif /* __MATMUL_HPP__ */
