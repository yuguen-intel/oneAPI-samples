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

      constexpr bool kShift = true;

      int rows_to_compute = k_size;
      while (rows_to_compute>1) {
        T rq[k_size][4];
        T previous_givens[2][2] = {{1, 0}, {0, 1}};

        T shift_value = T{0};
        if constexpr (kShift){

          // Compute the shift value
          // Take the submatrix:
          // [a b] 
          // [b c]
          // and compute the shift such as
          // mu = c - (sign(d)* b*b)/(abs(d) + sqrt(d*d + b*b))
          // where d = (a - c)/2
          T a = rows_to_compute-2 < 0 ? T{0} : a_tri_diag[rows_to_compute-2][1];
          T b = rows_to_compute-1 < 0 ? T{0} : a_tri_diag[rows_to_compute-1][0];
          T c = rows_to_compute-1 < 0 ? T{0} : a_tri_diag[rows_to_compute-1][1];

          T d = (a - c) / 2;
          T b_squared = b*b;
          T d_squared = d*d;
          T b_squared_signed = d<0 ? -b_squared : b_squared;
          shift_value = c - b_squared_signed / (abs(d) + sqrt(d_squared + b_squared));

          // Subtract the shift from the diagonal of RQ
          for (int row = 0; row < rows_to_compute; row++) {
            a_tri_diag[row][1] -= shift_value;
          }
        }

        PRINTF("a_tri_diag before QR\n");
        for(int row=0; row<k_size; row++){
          for(int col=0; col<4; col++){
            PRINTF("%f ", a_tri_diag[row][col]);
          }
          PRINTF("\n");
        }

        // Go through the rows by pairs
        for (int row = 0; row < rows_to_compute; row++) {
      // if (row == 0){
      //     PRINTF("a_tri_diag at i=%d\n", iteration);
      //     for(int row=0; row<k_size; row++){
      //       for(int col=0; col<4; col++){
      //         PRINTF("%f ", a_tri_diag[row][col]);
      //       }
      //       PRINTF("\n");
      //     }
      // }

          // Take the diagonal element and the one below it
          T x = a_tri_diag[row][1];
          T y = row + 1 > k_size - 1 ? T{0} : a_tri_diag[row + 1][0];

          // PRINTF ("Givens parameters %f %f\n", x, y);

          // Compute the Givens rotation matrix
          // // c = sqrt(x*x + y*y) / (x + y*y/x)
          // // s = -c * y/x
          // T x_squared = x*x;
          // T y_squared = y*y;
          // T c = sqrt(x_squared + y_squared) / (x + y_squared/x);
          // T s = -c*y/x;

          // c = x/sqrt(x*x + y*y)
          // s = -y/sqrt(x*x + y*y)
          T x_squared = x * x;
          T y_squared = y * y;
          T norm = sycl::sqrt(x_squared + y_squared);
          T sign = x >= 0 ? 1 : -1;
          T c = sign * x / norm;
          T s = sign * -y / norm;

          T givens[2][2];
          givens[0][0] = c;
          givens[0][1] = -s;
          givens[1][0] = s;
          givens[1][1] = c;

          // PRINTF ("Givens \n");
          // for (int i=0; i<2; i++){
          //   for (int j=0; j<2; j++){
          //     PRINTF ("%f ", givens[i][j]);
          //   }
          //   PRINTF ("\n");
          // }

          if (row < k_size - 1) {
            // Muliply the Givens rotation matrix with the appropriate rows of A
            for (int j = 0; j < 4; j++) {
              T a0 = (j == 3) ? T{0} : a_tri_diag[row][j + 1];
              T a1 = a_tri_diag[row + 1][j];
              for (int i = 0; i < 2; i++) {
                T dot_product = givens[i][0] * a0;
                dot_product += givens[i][1] * a1;

                // Handling the last row a buit differently to avoid doing on
                // extra iteration
                if(i + row < rows_to_compute){
                  if (i + row == k_size - 1) {
                    if (j > 0) {
                      a_tri_diag[row + i][j - 1] = -dot_product;
                    }
                  } else {
                    a_tri_diag[row + i][j] = dot_product;
                  }
                }

                if (i == 0) {
                  if (j + 1 > k_size - 1) {
                    rq[row][0] = 0;
                  } else {
                    rq[row][j + 1] = dot_product;
                  }
                } else if (row + 1 == k_size - 1) {
                  if (j == 1) {
                    rq[row + 1][j] = -dot_product;
                  } else {
                    rq[row + 1][j] = 0;
                  }
                }
              }
            }
          }
      // if (rows_to_compute-1 == row){

          PRINTF("a_tri_diag after QRD\n");
          for(int row=0; row<k_size; row++){
            for(int col=0; col<4; col++){
              PRINTF("%f ", a_tri_diag[row][col]);
            }
            PRINTF("\n");
          }

          // PRINTF("rq after QRD\n");
          // for(int row=0; row<k_size; row++){
          //   for(int col=0; col<4; col++){
          //     PRINTF("%f ", rq[row][col]);
          //   }
          //   PRINTF("\n");
          // }

      // }

          PRINTF("rq initialisation\n");
          for(int row=0; row<k_size; row++){
            for(int col=0; col<4; col++){
              PRINTF("%f ", rq[row][col]);
            }
            PRINTF("\n");
          }

          //   PRINTF ("Previous Givens \n");
          //   for (int i=0; i<2; i++){
          //     for (int j=0; j<2; j++){
          //       PRINTF ("%f ", previous_givens[i][j]);
          //     }
          //     PRINTF ("\n");
          //   }

          // PRINTF("row= %d\n", row);
          // Compute R*Q
          // Muliply the Givens rotation matrix with the appropriate rows of A
          for (int j = 0; j < 4; j++) {
            int current_row = j - 2 + row - 1;
            int col = 1 - j + 2;
            // PRINTF("Reading rq[%d][%d]\n", current_row, col);
            // PRINTF("Reading rq[%d][%d]\n", current_row, col+1);

            T a0 = current_row < 0 || current_row > k_size - 1 || col < 0 ||
                           col > k_size - 1
                       ? T{0}
                       : rq[current_row][col];
            T a1 = current_row < 0 || current_row > k_size - 1 || col + 1 < 0 ||
                           col + 1 > k_size - 1
                       ? T{0}
                       : rq[current_row][col + 1];
            for (int i = 0; i < 2; i++) {
              // PRINTF("computing rq %d %d\n", current_row, col+i);
              // PRINTF("taking %f %f\n", a0, previous_givens[0][i]);
              // PRINTF("taking %f %f\n", a1, previous_givens[1][i]);
              T dot_product = a0 * previous_givens[0][i];
              dot_product += a1 * previous_givens[1][i];
              // PRINTF("= %f\n", dot_product);
            PRINTF("Current row %d col %d\n", current_row, col+i);
            PRINTF("row %d rows_to_compute %d\n", row, rows_to_compute);

              if (current_row >= 0 && current_row < rows_to_compute && col + i >= 0 &&
                  col + i < 4 && row > 0) {
                bool last_row_last_elem = (current_row == k_size - 1) && (col + i == 1);
                bool previous_to_last_row_last_elem = (current_row == k_size - 2) && (col + i == 2);
                PRINTF("Writing rq %d %d\n", current_row, col+i);
                if (last_row_last_elem || previous_to_last_row_last_elem) {
                  rq[current_row][col + i] = -dot_product;
                } else {
                  rq[current_row][col + i] = dot_product;
                }
              }
            }
          }

          PRINTF("rq modified\n");
          for(int row=0; row<k_size; row++){
            for(int col=0; col<4; col++){
              PRINTF("%f ", rq[row][col]);
            }
            PRINTF("\n");
          }
            PRINTF("\n");
          //   PRINTF("\n");

          for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
              if (i == j) {
                previous_givens[i][j] = givens[i][j];
              } else {
                previous_givens[i][j] = -givens[i][j];
              }
            }
          }
        }

      // if (matrix_index == 26){
        // PRINTF("rq\n");
        // for(int row=0; row<k_size; row++){
        //   for(int col=0; col<4; col++){
        //     PRINTF("%f ", rq[row][col]);
        //   }
        //   PRINTF("\n");
        // }
      // }



        // Copy RQ in A
        for (int row = 0; row < rows_to_compute; row++) {
          for (int col = 0; col < 4; col++) {
            if (col == 3) {
              a_tri_diag[row][col] = T{0};
            } else {
              a_tri_diag[row][col] = rq[row][col];
            }
          }
        }

        PRINTF("rq \n");
        for(int row=0; row<k_size; row++){
          for(int col=0; col<4; col++){
            PRINTF("%f ", rq[row][col]);
          }
          PRINTF("\n");
        }



        if constexpr (kShift){
          // Add the shift back to the diagonal of RQ
          for (int row = 0; row < rows_to_compute; row++) {
            a_tri_diag[row][1] += shift_value;
          }
        }

        PRINTF("a_tri_diag (rq+shift %f)\n", shift_value);
        for(int row=0; row<k_size; row++){
          for(int col=0; col<4; col++){
            PRINTF("%f ", a_tri_diag[row][col]);
          }
          PRINTF("\n");
        }
        // check if condition is reached
        float constexpr threshold = 1e-3;
        if(sycl::fabs(rq[rows_to_compute-1][0]) < threshold){
          rows_to_compute--;
          PRINTF("REDUCING ROWS TO COMPUTE TO %d\n", rows_to_compute);
        }
     
        // if(rows_to_compute==1){
        //   PRINTF("a_tri_diag 0 0 %f\n", sycl::fabs(a_tri_diag[0][0]));
        //   if(sycl::fabs(a_tri_diag[rows_to_compute-1][0]) > threshold){
        //     rows_to_compute++;
        //   }
        // }

        // bool reached = true;
        // // PRINTF("===============================================================");
        // // PRINTF("checking... ");
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

      if(matrices_computed == 1024){
        PRINTF("Average number of iterations: %d\n", iterations/matrices_computed);
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