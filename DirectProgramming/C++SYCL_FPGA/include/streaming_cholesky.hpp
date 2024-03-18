#ifndef __STREAMING_CHOLESKY_HPP__
#define __STREAMING_CHOLESKY_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

using namespace sycl;

#define PRINTF(format, ...)                                    \
  {                                                            \
    static const CL_CONSTANT char _format[] = format;          \
    ext::oneapi::experimental::printf(_format, ##__VA_ARGS__); \
  }

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

namespace fpga_linalg {

/*
  Cholesky decomposition - Computes L such that A=LL* where:
  - A is the input matrix (hermitian, positive definite)
  - L is a lower triangular matrix
  - L* is the conjugate transpose of L

  This function implements a modified version of the Cholesky–Banachiewicz
  algorithm.
  Pseudo code:

  int row_size = 0;
  for (column = 0; column <= row_size; column++) {
    for (row = column; row < rows; row++) {
      float sum = 0;
      for (k = 0; k < column; k++)
        sum += L[row][k] * L[column][k];

      if (row == column)
        L[row][column] = sqrt(A[row][row] - sum);
      else
        L[row][column] = (A[row][column] - sum) / L[column][column];
    }
  }

  The input and output matrices are consumed/produced from/to pipes.
*/
template <typename T,       // The datatype for the computation
          bool is_complex,  // True if T is ac_complex<X>
          int rows,         // Number of rows==columns in the A matrices
          int raw_latency,  // Read after write (RAW) latency (in iterations) of
          int raw_latency_inversion,  // Read after write (RAW) latency (in iterations) of
                            // the triangular loop of this function.
                            // This value depends on the FPGA target, the
                            // datatype, the target frequency, etc.
                            // This value will have to be tuned for optimal
                            // performance. Refer to the Triangular Loop
                            // design pattern tutorial.
                            // In general, find a high value for which the
                            // compiler is able to achieve an II of 1 and
                            // go down from there.
          int pipe_size,    // Number of elements read/write per pipe operation
                            // to read the input matrix
          typename AIn,     // A matrix input pipe, receive pipe_size
                            // elements from the pipe with each read
          typename IOut     // I matrix output pipe, send one element to the
                            // pipe with each write.
          >
struct StreamingCholesky {
  void operator()() const {
    // Functional assertions
    static_assert(rows >= 4,
                  "Only matrices of size 4x4 and over are supported");
    static_assert(pipe_size >= 1,
                  "The pipe must be able to contain at least one element");

    // Set the computation type to T or ac_complex<T> depending on the value
    // of is_complex
    using TT = std::conditional_t<is_complex, ac_complex<T>, T>;

    constexpr int kColumns = rows;

    // Number of lower-left elements in the L output matrix
    constexpr int kLMatrixSize = kColumns * (kColumns + 1) / 2;

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
      TT a_load[rows][kColumns];

      // Two copies of L to be able to load two complete rows per iteration
      // Multiple private copies to be able to overlap multiple loop
      // iterations
      [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
      TT l_result_compute[rows][kColumns];
      [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
      TT l_result_compute_copy[rows][kColumns];

      [[intel::private_copies(2)]]  // NO-FORMAT: Attribute
      TT l_result[kLMatrixSize];

      // The compiler has difficulty automatically figuring out an optimal
      // configuration for these memories, so force all relevant parameters.
      // The number of private copies ensures the compiler will schedule
      // as many overlapping loop iterations as possible
      // The code is written so that each memory is single read/single write
      // so there is no need for any replicate.
      // L matrix read from pipe
      [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
      TT l_matrix[rows][kColumns];

      // L inverse matrix for the compute
      [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
      TT li_matrix_compute[rows][kColumns];

      // L inverse matrix
      [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
      TT li_matrix[rows][kColumns];

      // Final inverse matrix (only the triangular elements)
      [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
      TT i_matrix[kColumns * (kColumns + 1) / 2];

      // Copy a matrix from the pipe to a local memory
      // Number of pipe reads of pipe_size required to read a full column
      constexpr int kExtraIteration = ((rows % pipe_size) != 0) ? 1 : 0;
      constexpr int kLoopIterPerColumn = (rows / pipe_size) + kExtraIteration;
      // Number of pipe reads of pipe_size to read all the matrices
      constexpr int kLoopIter = kLoopIterPerColumn * kColumns;
      // Size in bits of the loop iterator over kLoopIter iterations
      constexpr int kLoopIterBitSize =
          fpga_tools::BitsForMaxValue<kLoopIter + 1>();

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        fpga_tools::NTuple<TT, pipe_size> pipe_read = AIn::read();

        int write_idx = li % kLoopIterPerColumn;

        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
            if constexpr (k * pipe_size + t < kColumns) {
              if (write_idx == k) {
                a_load[li / kLoopIterPerColumn][k * pipe_size + t] =
                    pipe_read.template get<t>();
              }
            }

            // Delay data signals to create a vine-based data distribution
            // to lower signal fanout.
            pipe_read.template get<t>() =
                sycl::ext::intel::fpga_reg(pipe_read.template get<t>());
          });

          write_idx = sycl::ext::intel::fpga_reg(write_idx);
        });
      }

/*
      // Computation of the number of iterations required for the triangular
      // loop. Refer to the triangular_loop tutorial for details on how
      // to compute these.
      constexpr int kRegularIterations = kColumns * (kColumns + 1) / 2;
      constexpr int kExtraIterations = (raw_latency - 1) * raw_latency / 2;
      constexpr int kExtraIterationsToRemove =
          kColumns >= raw_latency
              ? 0
              : (raw_latency - kColumns) * (raw_latency - kColumns + 1) / 2;
      constexpr int kTotalIterations =
          kRegularIterations + kExtraIterations - kExtraIterationsToRemove;

      // Compute the L matrix
      int column = 0;
      int row = 0;
      TT div_term{0};
      TT partial_sum_0{0};
      TT partial_sum_1{0};
      bool first = true;
      [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
      [[intel::ivdep(raw_latency*2)]] // NO-FORMAT: Attribute
      for (int iteration = 0; iteration < kTotalIterations*2; iteration++) {
        // Perform the dot product of the elements of the two rows indexed by
        // row and column from element 0 to column

        // TT sum = 0;
        // fpga_tools::UnrolledLoop<kColumns>([&](auto k) {
        //   TT to_add;
        //   bool should_compute =  k < column;
        //   TT mul_lhs = should_compute ? l_result_compute[row][k] : T{0};
        //   TT mul_rhs = should_compute ? l_result_compute_copy[column][k] : T{0};

        //   if constexpr (is_complex) {
        //     to_add = mul_lhs * mul_rhs.conj();
        //   } else {
        //     to_add = mul_lhs * mul_rhs;
        //   }
        //   sum += to_add;
        // });

        [[intel::fpga_register]]
        TT mul_lhs[kColumns];
        [[intel::fpga_register]]
        TT mul_rhs[kColumns];

        fpga_tools::UnrolledLoop<kColumns>([&](auto k) {
          bool should_compute =  k < column;
          mul_lhs[k] = should_compute ? l_result_compute[row][k] : T{0};
          mul_rhs[k] = should_compute ? l_result_compute_copy[column][k] : T{0};
        });


        TT sum = 0;
        fpga_tools::UnrolledLoop<kColumns/2>([&](auto k) {
          TT lhs, rhs;
          if(first){
            lhs = mul_lhs[k];
            rhs = mul_rhs[k].conj();
          }
          else{
            lhs = mul_lhs[k+(kColumns/2)];
            rhs = mul_rhs[k+(kColumns/2)].conj();
          }

          sum+=lhs * rhs;
        });

        if (first){
          partial_sum_0 = sum;
        }
        else {
          partial_sum_1 = sum;
        }
        TT full_sum = partial_sum_0 + partial_sum_1;


        TT a_loaded = (row < rows) ? a_load[row][column] : TT{0};
        TT diff = a_loaded - full_sum;

        // Only do useful work for meaningful iterations
        if (!first){

          TT to_store;
          if (row == column) {
            // Perform the reciprocal sqrt rather than the sqrt because:
            // - it has a shorter latency and will reduce the RAW latency
            //   of the loop
            // - the result of the sqrt is used as a divisor which is also
            //   a long operation, so replacing x/sqrt by x*rsqrt will save
            //   latency
            // - the diagonal elements will need to be inverted later, but we
            //   can do that while outside this loop when we transfer the L
            //   matrix to the pipe
            if constexpr (is_complex) {
              div_term = {sycl::rsqrt(diff.r()), 0};
            } else {
              div_term = sycl::rsqrt(diff);
            }
            to_store = div_term;
          } else {
            to_store = diff * div_term;
          }

          if (column <= row) {
            // Store the results to two working copies of L to be able to read
            // two complete rows at each iteration
            l_result_compute[row][column] = to_store;
            l_result_compute_copy[row][column] = to_store;
            // Store the result to the output matrix
            if constexpr (is_complex) {
              l_result[row * (row + 1) / 2 + column] = to_store.conj();
            } else {
              l_result[row * (row + 1) / 2 + column] = to_store;
            }
          }
        }

        // Update loop indexes
        if (!first) {
          if (row == (rows - 1)) {
            column = column + 1;
            row = sycl::min(column, rows - raw_latency);
          } else {
            row = row + 1;
          }
        }
        
        first = !first;

      }  // end of iteration

      // Go over the L matrix and write each element to the pipe
      int l_idx = 0;
      [[intel::loop_coalesce(2)]] // NO-FORMAT: Attribute
      for (int row = 0; row < rows; row++) {
        for (int column = 0; column <= row; column++) {
          TT to_write;
          TT current_l_value = l_result[l_idx];
          // The diagonal elements need to be inverted as the
          // inversion was removed from the above compute loop
          // to reduce the RAW latency
          if (row == column) {
            if constexpr (is_complex) {
              to_write = {1 / current_l_value.r(), 0};
            }
            else{
              to_write = 1 / current_l_value;
            }

          } else {
            to_write = current_l_value;
          }

          
          l_matrix[row][column] = to_write.conj();


          l_idx++;
        }
      }

*/
      // Count the total number of loop iterations, using the triangular loop
      // optimization (refer to the Triangular Loop design pattern tutorial)
      constexpr int kNormalIterations = kColumns * (kColumns + 1) / 2;
      constexpr int kExtraIterationsInversion =
          (raw_latency_inversion > rows) ? ((rows - 1) * (raw_latency_inversion - rows)) +
                                     ((rows - 2) * (rows - 1)) / 2
                               : (raw_latency_inversion - 2) * (raw_latency_inversion - 2 + 1) / 2;
      constexpr int kTotalIterationsInversion = kNormalIterations + kExtraIterationsInversion;

      // All the loop control variables with all the requirements to apply
      // some shannonization (refer to the Shannonization tutorial)

      int diagonal_number = 0;
      int next_diagonal_number = 1;
      int diagonal_size = (kColumns > raw_latency_inversion ? kColumns : raw_latency_inversion) - 1;
      int col = diagonal_number;
      int row_inversion = 0;

      TT inv_partial_sum_0{0};
      TT inv_partial_sum_1{0};
      bool first_inversion{true};
      TT div_val{0};

      [[intel::ivdep(raw_latency_inversion*2 - 1)]]  // NO-FORMAT: Attribute
      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (int it = 0; it < kTotalIterationsInversion*2; it++) {
        // Only perform work when in not dummy iterations
        if ((row_inversion < rows) & (col < kColumns)) {

          TT col_reg[kColumns];
          fpga_tools::UnrolledLoop<kColumns>([&](auto k) {
            // col_reg[k] = l_matrix[col][k];
            col_reg[k] = a_load[col][k];
          });


          TT partial_sum{0};
          fpga_tools::UnrolledLoop<kColumns/2>([&](auto k) {
            auto li_loaded = first_inversion ? col_reg[k] : col_reg[k + (kColumns/2)];

            bool check_lhs = (first_inversion && (k > col)) || (!first_inversion && ((k + (kColumns/2)) > col));
            TT lhs;
            if (check_lhs) {
              lhs = TT{0};
            } else {
              lhs = li_loaded;
            }




            bool rhs_check = (first_inversion && (k >= row_inversion) && (k < col)) ||
                             (!first_inversion && ((k + (kColumns/2)) >= row_inversion) && ((k + (kColumns/2)) < col));

            TT rhs;
            if (rhs_check) {
              TT li_compute_load;
              if (first_inversion){
                li_compute_load = li_matrix_compute[row_inversion][k];
              //   PRINTF("it: %d -> Reading row %d col %d\n", it, row_inversion, (int) k);
              }
              else {
                li_compute_load = li_matrix_compute[row_inversion][k + (kColumns/2)];
              //   PRINTF("it: %d -> Reading row %d col %d\n", it, row_inversion, (int) (k + (kColumns/2)));          
              }
              rhs = li_compute_load;
            } else {
              rhs = TT{0};
            }

            bool div_check = (first_inversion && (k == col)) || (!first_inversion && ((k + (kColumns/2)) == col));
            if (div_check) {
              div_val = lhs;
            }

            partial_sum -= lhs * rhs;
          });


          if (first_inversion) {
            inv_partial_sum_0 = partial_sum;
          }
          else {
            inv_partial_sum_1 = partial_sum;
          }

          TT init_sum_value = (row_inversion == col) ? TT{1} : TT{0};
          TT current_sum = init_sum_value + inv_partial_sum_0 + inv_partial_sum_1;
          TT result = current_sum / div_val;


          if (!first_inversion){

            // Write the result to both the working copy and the final matrix
            // This is done to only have matrices with a single read and a
            // single write.
            li_matrix_compute[row_inversion][col] = result;
            li_matrix[row_inversion][col] = result;
            // PRINTF("it: %d -> Writing row %d col %d\n", it, row_inversion, col);          

          }
        }

        if (!first_inversion){
          if (row_inversion == diagonal_size) {
            diagonal_number = next_diagonal_number;
            diagonal_size =
                std::max(kColumns - next_diagonal_number, raw_latency_inversion) - 1;
            col = next_diagonal_number;
            row_inversion = 0;
            next_diagonal_number++;
          } else {
            row_inversion++;
            col++;
          }
        }
        first_inversion = !first_inversion;
      }

      int inverse_matrix_read_idx = 0;
      for (int loop_count = 0; loop_count < kNormalIterations; loop_count++) {
        IOut::write(li_matrix[inverse_matrix_read_idx/rows][inverse_matrix_read_idx%rows]);
        inverse_matrix_read_idx++;
      }

/*
      int inverse_matrix_write_idx = 0;
      // Compute inv(A) = inv(L)*trans(inv(L))
      for (int col = 0; col < rows; col++) {
        TT col_of_transpose_matrix[rows];
        int row_index;

        TT partial_sum_0{0};
        TT partial_sum_1{0};
        bool first = true;
        int row_counter_adjusted = col;
        for (int row_counter = col; row_counter < rows + (rows-col); row_counter++) {
          if (row_counter_adjusted >= rows) {
            row_index = row_counter_adjusted - rows;
          } else {
            row_index = row_counter_adjusted;
          }

          TT row[kColumns];
          fpga_tools::UnrolledLoop<kColumns>([&](auto k) {
            row[k] = li_matrix[row_index][k];
          });


          TT partial_sum{0};
          fpga_tools::UnrolledLoop<kColumns/2>([&](auto k) {
            TT li_load = first ? row[k] : row[k + (kColumns/2)];

            if (row_index == col) {
              if (first){
                col_of_transpose_matrix[k] = li_load;
              }
              else{
                col_of_transpose_matrix[k + (kColumns/2)] = li_load;
              }
            }

            bool check_row_index = first ? k < row_index : (k + (kColumns/2)) < row_index;
            auto lhs = check_row_index ? TT{0} : li_load;

            TT rhs;
            bool check_col = first ? k < col : (k + (kColumns/2)) < col;
            if (check_col) {
              rhs = TT{0};
            }
            else if (first) {
              rhs = col_of_transpose_matrix[k];
            }
            else {
              rhs = col_of_transpose_matrix[k + (kColumns/2)];
            }

            partial_sum += lhs * rhs.conj();
          });

          if (first){
            partial_sum_0 = partial_sum;
          }
          else {
            partial_sum_1 = partial_sum;
          }

          TT elem = partial_sum_0 + partial_sum_1;
          if (!first) {
            i_matrix[inverse_matrix_write_idx] = elem;
            inverse_matrix_write_idx++;
          }

          if(!first) {
            row_counter_adjusted++;
          }
          first = !first;
        }
      }

      int inverse_matrix_read_idx = 0;
      for (int loop_count = 0; loop_count < kNormalIterations; loop_count++) {
        IOut::write(i_matrix[inverse_matrix_read_idx]);
        inverse_matrix_read_idx++;
      }
*/
    }  // end of while(1)
  }    // end of operator
};     // end of struct

}  // namespace fpga_linalg

#endif /* __STREAMING_CHOLESKY_HPP__ */