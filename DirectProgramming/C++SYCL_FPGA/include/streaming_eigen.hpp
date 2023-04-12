#ifndef __STREAMING_QRD_HPP__
#define __STREAMING_QRD_HPP__

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

namespace fpga_linalg {

/*
  QRD (QR decomposition) - Computes Q and R matrices such that A=QR where:
  - A is the input matrix
  - Q is a unitary/orthogonal matrix
  - R is an upper triangular matrix

  This function implements a OneAPI optimized version of the "High performance
  QR Decomposition for FPGAs" FPGA'18 paper by Martin Langhammer and Bogdan
  Pasca.

  Each matrix (input and output) are represented in a column wise (transposed).

  Then input and output matrices are consumed/produced from/to pipes.
*/
template <typename T,          // The datatype for the computation
          int k_size,          // Number of rows/columns in the A matrices
          int pipe_size,       // Number of elements read/write per pipe
                               // operation
          typename AIn,        // A matrix input pipe, receive pipe_size
                               // elements from the pipe with each read
          typename QOut,       // Q matrix output pipe, send pipe_size
                               // elements to the pipe with each write
          typename EValuesOut  // R matrix output pipe, send pipe_size
                               // elements to the pipe with each write.
                               // Only upper-right elements of R are
                               // sent in row order, starting with row 0.
          >
struct StreamingEigen {
  void operator()() const {
    // Functional limitations
    static_assert(k_size >= 4,
                  "only matrices of size 4x4 and over are supported");

    // Type used to store the matrices in the compute loop
    using column_tuple = fpga_tools::NTuple<T, k_size>;

    // Count the total number of iterations performed to compute the average
    int iterations = 0;
    int matrices_computed = 0;

    // Compute Eigen values as long as matrices are given as inputs
    while (1) {
      // Three copies of the full matrix, so that each matrix has a single
      // load and a single store.
      // a_load is the initial matrix received from the pipe
      // a_compute is used and modified during calculations
      // q_result is a copy of a_compute and is used to send the final output

      // Break memories up to store 4 complex numbers (32 bytes) per bank
      constexpr short kBankwidth = pipe_size * sizeof(T);
      constexpr unsigned short kNumBanks = k_size / pipe_size;

      // When specifying numbanks for a memory, it must be a power of 2.
      // Unused banks will be automatically optimized away.
      constexpr short kNumBanksNextPow2 =
          fpga_tools::Pow2(fpga_tools::CeilLog2(kNumBanks));

      [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      column_tuple a_load[k_size],
          a_compute[k_size], q_result[k_size];

      // Contains the values of the upper-right part of R in a row by row
      // fashion, starting by row 0
      [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
      T eigen_values[k_size];

      // Copy a matrix from the pipe to a local memory
      // Number of pipe reads of pipe_size required to read a full column
      constexpr int kExtraIteration = (k_size % pipe_size) != 0 ? 1 : 0;
      constexpr int kLoopIterPerColumn = k_size / pipe_size + kExtraIteration;
      // Number of pipe reads of pipe_size to read all the matrices
      constexpr int kLoopIter = kLoopIterPerColumn * k_size;
      // Size in bits of the loop iterator over kLoopIter iterations
      constexpr int kLoopIterBitSize =
          fpga_tools::BitsForMaxValue<kLoopIter + 1>();

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        fpga_tools::NTuple<T, pipe_size> pipe_read = AIn::read();

        int write_idx;
        int a_col_index;
        write_idx = li % kLoopIterPerColumn;
        a_col_index = li / kLoopIterPerColumn;

        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
            if (write_idx == k) {
              if constexpr (k * pipe_size + t < k_size) {
                a_load[a_col_index].template get<k * pipe_size + t>() =
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

      // Compute the Eigen values

      // store the entire matrix in a 3 x k_size table as the matrix is
      // tridiagonal, and we'll need a fourth diagonal for R
      T a_tri_diag[k_size][4];
      for (int col = 0; col < 4; col++) {
        fpga_tools::UnrolledLoop<k_size>([&](auto k) {
          if constexpr (k == 0) {
            if (col == 0) {
              a_tri_diag[k][col] = T{0};
            } else {
              a_tri_diag[k][col] = a_load[col + k - 1].template get<k>();
            }
          } else {
            if ((col + k) > k_size) {
              a_tri_diag[k][col] = T{0};
            }
            a_tri_diag[k][col] = a_load[col + k - 1].template get<k>();
          }
        });
      }

      // PRINTF("a_tri_diag\n");
      // for(int row=0; row<k_size; row++){
      //   for(int col=0; col<4; col++){
      //     PRINTF("%f ", a_tri_diag[row][col]);
      //   }
      //   PRINTF("\n");
      // }

      constexpr bool kShift = false;

      int rows_to_compute = k_size;
      while (rows_to_compute > 1) {
      // while (rows_to_compute > 1) {
        PRINTF("========================================================\n");
        T rq[k_size][4];

        for (int row = 0; row < k_size; row++) {
          for (int col = 0; col < 4; col++) {
            rq[row][col] = -1;
          }
        }

        T givens_it_minus_1[2][2] = {{1, 0}, {0, 1}};
        T givens_it_minus_2[2][2] = {{1, 0}, {0, 1}};

        T a_result_rq_buffer[3][4];

        for (int row = 0; row < 3; row++) {
          for (int col = 0; col < 4; col++) {
            a_result_rq_buffer[row][col] = -66;
          }
        }

        T shift_value = T{0};
        if constexpr (kShift) {
          // Compute the shift value
          // Take the submatrix:
          // [a b]
          // [b c]
          // and compute the shift such as
          // mu = c - (sign(d)* b*b)/(abs(d) + sqrt(d*d + b*b))
          // where d = (a - c)/2
          T a = rows_to_compute - 2 < 0 ? T{0}
                                        : a_tri_diag[rows_to_compute - 2][1];
          T b = rows_to_compute - 1 < 0 ? T{0}
                                        : a_tri_diag[rows_to_compute - 1][0];
          T c = rows_to_compute - 1 < 0 ? T{0}
                                        : a_tri_diag[rows_to_compute - 1][1];

          T d = (a - c) / 2;
          T b_squared = b * b;
          T d_squared = d * d;
          T b_squared_signed = d < 0 ? -b_squared : b_squared;
          shift_value =
              c - b_squared_signed / (abs(d) + sqrt(d_squared + b_squared));

          // Subtract the shift from the diagonal of RQ
          for (int row = 0; row < rows_to_compute; row++) {
            a_tri_diag[row][1] -= shift_value;
          }
        }

        // PRINTF("a_tri_diag before QR\n");
        // for (int row = 0; row < k_size; row++) {
        //   for (int col = 0; col < 4; col++) {
        //     PRINTF("%f ", a_tri_diag[row][col]);
        //   }
        //   PRINTF("\n");
        // }

        T last_rq_val = -55;

        for (int row = 0; row < k_size + 1; row++) {
          // PRINTF("-------------------------------------------------------\n");

          T givens[2][2];

          if (row < k_size) {
            // Take the diagonal element and the one below it
            T x = a_tri_diag[row][1];
            T y = row + 1 > k_size - 1 ? T{0} : a_tri_diag[row + 1][0];

            // Compute the Givens rotation matrix
            T x_squared = x * x;
            T y_squared = y * y;
            T norm = sycl::sqrt(x_squared + y_squared);
            T sign = x >= 0 ? 1 : -1;
            T c = sign * x / norm;
            T s = sign * -y / norm;

            // The final Givens matrix must be (-1 0, 0 1)
            // to get the correct sign for the final value of A

            givens[0][0] = row == (k_size - 1) ? -1 : c;
            givens[0][1] = row == (k_size - 1) ? 0 : -s;
            givens[1][0] = row == (k_size - 1) ? 0 : s;
            givens[1][1] = row == (k_size - 1) ? 1 : c;

            // PRINTF("Givens \n");
            // for (int i = 0; i < 2; i++) {
            //   for (int j = 0; j < 2; j++) {
            //     PRINTF("%f ", givens[i][j]);
            //   }
            //   PRINTF("\n");
            // }

            /*
              Compute the sub product givens*A
              This only affect two rows of A
              Here with the example of the first givens matrix
                         (a00 a01 a02 a03)
                         (a10 a11 a12 a13)
                  x      (a20 a21 a22 a23)
                         (a30 a31 a32 a33)

              (c -s 0 0) (a0  b0  c0  d0 )
              (s  c 0 0) (a1  b1  c1  d1 )
              (0  0 1 0) (a20 a21 a22 a23)
              (0  0 0 1) (a30 a31 a32 a33)

            */
            for (int j = 0; j < 4; j++) {
              // Go through all the columns of A
              // Because A is tridiagonal, there are only 3 columns, but we
              // need one more for intermediate results

              // Take the first two elements of the current column of A
              // Continuing with the previous example: a00 and a10
              T a0 = (j == 3) ? T{0} : a_tri_diag[row][j + 1];
              T a1 = (row + 1) >= k_size ? T{0} : a_tri_diag[row + 1][j];

              // Go through the two affected rows
              for (int i = 0; i < 2; i++) {
                T new_a_element = givens[i][0] * a0 + givens[i][1] * a1;
                // Don't write beyond the last matrix row
                if (row != (k_size - 1) || (i == 0)) {
                  a_tri_diag[row + i][j] = new_a_element;
                }

                if (i == 0) {
                  a_result_rq_buffer[2][j] = new_a_element;
                }

              }  // end of i

            }  // end of j

          }  // end of if (row < k_size)

          // PRINTF("a_result_rq_buffer\n");
          // for (int row = 0; row < 3; row++) {
          //   for (int col = 0; col < 4; col++) {
          //     PRINTF("%f ", a_result_rq_buffer[row][col]);
          //   }
          //   PRINTF("\n");
          // }

          // PRINTF("Current givens \n");
          // for (int i = 0; i < 2; i++) {
          //   for (int j = 0; j < 2; j++) {
          //     PRINTF("%f ", givens[i][j]);
          //   }
          //   PRINTF("\n");
          // }

          // PRINTF("Givens m1\n");
          // for (int i = 0; i < 2; i++) {
          //   for (int j = 0; j < 2; j++) {
          //     PRINTF("%f ", givens_it_minus_1[i][j]);
          //   }
          //   PRINTF("\n");
          // }

          // PRINTF("Givens m2\n");
          // for (int i = 0; i < 2; i++) {
          //   for (int j = 0; j < 2; j++) {
          //     PRINTF("%f ", givens_it_minus_2[i][j]);
          //   }
          //   PRINTF("\n");
          // }

          if (row > 0) {
            int rq_col = row - 1;
            T g_prod_col[3] = {
                givens_it_minus_2[0][1] * givens_it_minus_1[0][0],
                givens_it_minus_2[1][1] * givens_it_minus_1[0][0],
                givens_it_minus_1[1][0]};

            // PRINTF("g_prod_col: %f %f %f\n", g_prod_col[0], g_prod_col[1],
            //        g_prod_col[2]);

            // PRINTF("RQ value at %d: %f \n", rq_col, last_rq_val);
            T dot_prod_val[2];
            T last_dp_val;
            for (int rq_row = 0; rq_row < 2; rq_row++) {
              T value = 0;
              // PRINTF("value = 0");
#pragma unroll
              for (int k = 0; k < 3; k++) {
                T g_prod_col_val =
                    (k + rq_row + 1) > 2 ? T{0} : g_prod_col[k + rq_row + 1];
                T a_result_rq_buffer_val = a_result_rq_buffer[rq_row + 1][k];
                value += a_result_rq_buffer_val * g_prod_col_val;
                // PRINTF(" + (%f * %f)", a_result_rq_buffer_val, g_prod_col_val);
              }
              // PRINTF(" = %f\n", value);
              dot_prod_val[rq_row] = value;
              last_dp_val = value;
            }

            for (int row_rq = 0; row_rq < k_size; row_rq++) {
              T rq_value;
              if (row_rq == rq_col) {
                rq_value = dot_prod_val[0];
              } else if (row_rq == (rq_col + 1)) {
                rq_value = dot_prod_val[1];
              } else if (row_rq == (rq_col - 1)) {
                rq_value = last_rq_val;
              } else {
                rq_value = 0;
              }

              int col_index = rq_col+1-row_rq;
              if (col_index < 4 && col_index >= 0){
                // PRINTF("Writing RQ %d %d = %f\n", row_rq, col_index, rq_value);
                rq[row_rq][col_index] = rq_value;
              }
            }

            last_rq_val = last_dp_val;
          }


          // PRINTF("a_tri_diag at row=%d\n", row);
          // for (int row = 0; row < k_size; row++) {
          //   for (int col = 0; col < 4; col++) {
          //     PRINTF("%f ", a_tri_diag[row][col]);
          //   }
          //   PRINTF("\n");
          // }

          // PRINTF("rq at row=%d\n", row);
          // for (int row = 0; row < k_size; row++) {
          //   for (int col = 0; col < 4; col++) {
          //     PRINTF("%f ", rq[row][col]);
          //   }
          //   PRINTF("\n");
          // }

          for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 4; col++) {
              a_result_rq_buffer[row][col] = a_result_rq_buffer[row + 1][col];
            }
          }

          // Store the previous Givens matrix for the computation of RQ
          // Also transpose this matrix
          givens_it_minus_2[0][0] = givens_it_minus_1[0][0];
          givens_it_minus_2[0][1] = givens_it_minus_1[0][1];
          givens_it_minus_2[1][0] = givens_it_minus_1[1][0];
          givens_it_minus_2[1][1] = givens_it_minus_1[1][1];

          givens_it_minus_1[0][0] = givens[0][0];
          givens_it_minus_1[0][1] = -givens[0][1];
          givens_it_minus_1[1][0] = -givens[1][0];
          givens_it_minus_1[1][1] = givens[1][1];

        }  // end of row

        // exit(0);


        // Copy RQ in A
        for (int row = 0; row < k_size; row++) {
          for (int col = 0; col < 4; col++) {
            bool is_first_elem = (row == 0) && (col == 0); 
            bool is_last_elem = (row == k_size-1) && (col == 2); 
            bool is_last_col = col == 3; 
            if ( is_first_elem || is_last_elem || is_last_col ) {
              a_tri_diag[row][col] = T{0};
            } else {
              a_tri_diag[row][col] = rq[row][col];
            }
          }
        }

        // PRINTF("rq \n");
        // for(int row=0; row<k_size; row++){
        //   for(int col=0; col<4; col++){
        //     PRINTF("%f ", rq[row][col]);
        //   }
        //   PRINTF("\n");
        // }

        if constexpr (kShift) {
          // Add the shift back to the diagonal of RQ
          for (int row = 0; row < rows_to_compute; row++) {
            a_tri_diag[row][1] += shift_value;
          }
        }

        // PRINTF("a_tri_diag (rq+shift %f)\n", shift_value);
        // for(int row=0; row<k_size; row++){
        //   for(int col=0; col<4; col++){
        //     PRINTF("%f ", a_tri_diag[row][col]);
        //   }
        //   PRINTF("\n");
        // }
        // check if condition is reached
        float constexpr threshold = 1e-3;
        if (sycl::fabs(rq[rows_to_compute - 1][0]) < threshold) {
          rows_to_compute--;
        }

        // if(rows_to_compute==1){
        //   PRINTF("a_tri_diag 0 0 %f\n", sycl::fabs(a_tri_diag[0][0]));
        //   if(sycl::fabs(a_tri_diag[rows_to_compute-1][0]) > threshold){
        //     rows_to_compute++;
        //   }
        // }

        // bool reached = true;
        // // //
        // // PRINTF("===============================================================");
        // // // PRINTF("checking... ");
        // for (int row = 1; row < k_size; row++) {
        //   // PRINTF("%f ", fabs(rq[row][0]));
        //   reached &= sycl::fabs(rq[row][0]) < threshold;
        // }
        // // PRINTF("\n");
        // cond = reached;
        // cond = iteration==0;
        iterations++;
        // if(iterations==2){
        //   PRINTF("NEEEEEEEEXXXXXXXXXXXXXXXXXXXXTTTT\n");
        //   exit(0);
        // }
        // if(iteration>10000){
        //   PRINTF("TOO MANY ITERATIONS!!!\n");
        //   for (int row = 0; row < k_size; row++) {
        //     PRINTF("%f\n", sycl::fabs(rq[row][0]));
        //   }
        // }
      }

      matrices_computed++;

      if (matrices_computed == 1024) {
        PRINTF("Average number of iterations: %d\n",
               iterations / matrices_computed);
      }

      // PRINTF("a_tri_diag\n");
      // for (int row = 0; row < k_size; row++) {
      //   for (int col = 0; col < 4; col++) {
      //     PRINTF("%f ", a_tri_diag[row][col]);
      //   }
      //   PRINTF("\n");
      // }

      for (int row = 0; row < k_size; row++) {
        eigen_values[row] = a_tri_diag[row][1];
      }

      // PRINTF("iteration %d\n", iteration);

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (int r_idx = 0; r_idx < k_size; r_idx++) {
        EValuesOut::write(eigen_values[r_idx]);
      }

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        int column_iter = li % kLoopIterPerColumn;
        bool get[kLoopIterPerColumn];
        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          get[k] = column_iter == k;
          column_iter = sycl::ext::intel::fpga_reg(column_iter);
        });

        fpga_tools::NTuple<T, pipe_size> pipe_write;
        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto t) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto k) {
            if constexpr (t * pipe_size + k < k_size) {
              pipe_write.template get<k>() =
                  get[t] ? q_result[li / kLoopIterPerColumn]
                               .template get<t * pipe_size + k>()
                         : sycl::ext::intel::fpga_reg(
                               pipe_write.template get<k>());
            }
          });
        });
        QOut::write(pipe_write);
      }

    }  // end of while(1)
  }    // end of operator
};     // end of struct

}  // namespace fpga_linalg

#endif /* __STREAMING_QRD_HPP__ */