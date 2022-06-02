#ifndef __STREAMING_MATMUL_HPP__
#define __STREAMING_MATMUL_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

namespace fpga_linalg {

/*
  Matrix mutliplication
*/
template <typename T,       // The datatype for the computation
          bool is_complex,  // True if T is ac_complex<X>
          int rows,         // Number of rows in the input matrices
          int columns,      // Number of columns in the input matrices
          int pipe_size,    // Number of elements read/write per pipe operation
                            // to read the input matrix
          typename AIn,     // A matrix input pipe, receive pipe_size
                            // elements from the pipe with each read
          typename BIn,     // B matrix input pipe, receive pipe_size
                            // elements from the pipe with each read
          typename MMOut    // MM matrix output pipe, send one elements to the
                            // pipe with each write.
          >
struct StreamingMatmul {
  void operator()() const {
    // Functional assertions
    static_assert(rows >= 4,
                  "Only matrices of size 4x4 and over are supported");
    static_assert(rows == columns,
                  "Only written for square matrices, can be extended");
    static_assert(pipe_size >= 1,
                  "The pipe must be able to contain at least one element");

    // Set the computation type to T or ac_complex<T> depending on the value
    // of is_complex
    using TT = std::conditional_t<is_complex, ac_complex<T>, T>;

    // Compute Cholesky decompositions as long as matrices are given as inputs
    while (1) {
      // Break memories up to store pipe_size elements per bank
      constexpr short kBankwidth = pipe_size * sizeof(TT);
      constexpr unsigned short kNumBanks = rows / pipe_size;

      // When specifying numbanks for a memory, it must be a power of 2.
      // Unused banks will be automatically optimized away.
      constexpr short kNumBanksNextPow2 =
          fpga_tools::Pow2(fpga_tools::CeilLog2(kNumBanks));

      [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      TT a_load[rows][columns];

      [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      TT b_load[rows][columns];

      [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      TT mm_result[rows][columns];

      // Copy a matrix from the pipe to a local memory
      // Number of pipe reads of pipe_size required to read a full column
      constexpr int kExtraIteration = ((rows % pipe_size) != 0) ? 1 : 0;
      constexpr int kLoopIterPerColumn = (rows / pipe_size) + kExtraIteration;
      // Number of pipe reads of pipe_size to read all the matrices
      constexpr int kLoopIter = kLoopIterPerColumn * columns;
      // Size in bits of the loop iterator over kLoopIter iterations
      constexpr int kLoopIterBitSize =
          fpga_tools::BitsForMaxValue<kLoopIter + 1>();

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        fpga_tools::NTuple<TT, pipe_size> pipe_read_a = AIn::read();
        fpga_tools::NTuple<TT, pipe_size> pipe_read_b = BIn::read();

        int write_idx = li % kLoopIterPerColumn;

        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
            if constexpr (k * pipe_size + t < columns) {
              if (write_idx == k) {
                a_load[li / kLoopIterPerColumn][k * pipe_size + t] =
                    pipe_read_a.template get<t>();
                b_load[li / kLoopIterPerColumn][k * pipe_size + t] =
                    pipe_read_b.template get<t>();
              }
            }

            // Delay data signals to create a vine-based data distribution
            // to lower signal fanout.
            pipe_read_a.template get<t>() =
                sycl::ext::intel::fpga_reg(pipe_read_a.template get<t>());
            pipe_read_b.template get<t>() =
                sycl::ext::intel::fpga_reg(pipe_read_b.template get<t>());
          });

          write_idx = sycl::ext::intel::fpga_reg(write_idx);
        });
      }

      // Compute the matrix product
      for (int row = 0; row < rows; row++) {
        for (int column = 0; column < columns; column++) {
          TT dot_prod{0};
          fpga_tools::UnrolledLoop<columns>([&](auto k) {
            // Assumes the B matrix was given transposed, otherwise it need to
            // be transposed.
            dot_prod = sycl::ext::intel::fpga_reg(dot_prod) +
                       a_load[row][k] * b_load[column][k];
          });
          mm_result[row][column] = dot_prod;
        }
      }

      // Copy the result matrix on the output pipe
      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        int column_iter = li % kLoopIterPerColumn;
        bool get[kLoopIterPerColumn];
        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          get[k] = column_iter == k;
          column_iter = sycl::ext::intel::fpga_reg(column_iter);
        });

        fpga_tools::NTuple<TT, pipe_size> pipe_write;
        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto t) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto k) {
            if constexpr (t * pipe_size + k < rows) {
              pipe_write.template get<k>() =
                  get[t] ? mm_result[li / kLoopIterPerColumn][t * pipe_size + k]
                         : sycl::ext::intel::fpga_reg(
                               pipe_write.template get<k>());
            }
          });
        });
        MMOut::write(pipe_write);
      }

    }  // end of while(1)
  }    // end of operator
};     // end of struct

}  // namespace fpga_linalg

#endif /* __STREAMING_MATMUL_HPP__ */