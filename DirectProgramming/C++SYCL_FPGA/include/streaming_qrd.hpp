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

  This function implements a oneAPI optimized version of the "High performance
  QR Decomposition for FPGAs" FPGA'18 paper by Martin Langhammer and Bogdan
  Pasca.

  Each matrix (input and output) are represented in a column wise (transposed).

  Then input and output matrices are consumed/produced from/to pipes.
*/
template <typename T,       // The datatype for the computation
          bool is_complex,  // True if T is ac_complex<X>
          int rows,         // Number of rows in the A matrices
          int columns,      // Number of columns in the A matrices
                            // , must be <= rows
          int raw_latency,  // Read after write latency (in iterations) of
                            // the triangular loop of this function.
                            // This value depends on the FPGA target, the
                            // datatype, the target frequency, etc.
                            // This value will have to be tuned for optimal
                            // performance. Refer to the Triangular Loop
                            // design pattern tutorial.
                            // In general, find a high value for which the
                            // compiler is able to achieve an II of 1 and
                            // go down from there.
          int pipe_size,    // Number of elements read/write per pipe
                            // operation
          typename AIn,     // A matrix input pipe, receive pipe_size
                            // elements from the pipe with each read
          typename QOut,    // Q matrix output pipe, send pipe_size
                            // elements to the pipe with each write
          typename ROut,    // R matrix output pipe, send pipe_size
                            // elements to the pipe with each write.
                            // Only upper-right elements of R are
                            // sent in row order, starting with row 0.
          bool k_column_order =
              true  // Default value is true for standard matrix input reads
                    // (reads the matrix one column at a time). False if read
                    // order by rows (sweeps the rows by pipe size). Each read
                    // contains pipe_size samples from the same column, then the
                    // next read contains samples from the next column.
          >
struct StreamingQRD {
  void operator()() const {
    // Functional limitations
    static_assert(rows >= columns,
                  "only rectangular matrices with rows>=columns are supported");
    static_assert(columns >= 4,
                  "only matrices of size 4x4 and over are supported");

    /*
      This code implements a oneAPI optimized variation of the following
      algorithm

      for i=0:n
        for j=max(i,1):n

          if(j==i)
            Q_i = a_i*ir
          else
            if(i>=0)
              a_j = a_j - s[j]*a_i

            if j=i+1
              pip1         = <a_{i+1},a_{i+1}>
              ir           = 1/sqrt(pip1)
              R_{i+1,i+1}  = sqrt(pip1)
            else
              p            = <a_{i+1}, a_j>
              s[j]         = p/pip1
              R_{i+1,j}    = p*ir


      Where:
      -> X_i represents the column i of the matrix X
      -> <x,y> represents the dot product of the vectors x and y
    */

    // Set the computation type to T or ac_complex<T> depending on the value
    // of is_complex
    using TT = std::conditional_t<is_complex, ac_complex<T>, T>;

    // Type used to store the matrices in the compute loop
    using column_tuple = fpga_tools::NTuple<TT, rows>;

    // Number of upper-right elements in the R output matrix
    constexpr int kRMatrixSize = columns * (columns + 1) / 2;

    // Compute QRDs as long as matrices are given as inputs
    while (1) {
      // Three copies of the full matrix, so that each matrix has a single
      // load and a single store.
      // a_load is the initial matrix received from the pipe
      // a_compute is used and modified during calculations
      // q_result is a copy of a_compute and is used to send the final output

      // Break memories up to store 4 complex numbers (32 bytes) per bank
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
      column_tuple a_load[columns], a_compute[columns], q_a_result[columns];

      // Contains the values of the upper-right part of R in a row by row
      // fashion, starting by row 0
      [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
      TT r_a_result[kRMatrixSize];

      // Copy a matrix from the pipe to a local memory
      // Number of pipe reads of pipe_size required to read a full column
      constexpr int kExtraIteration = (rows % pipe_size) != 0 ? 1 : 0;
      constexpr int kLoopIterPerColumn = rows / pipe_size + kExtraIteration;
      // Number of pipe reads of pipe_size to read all the matrices
      constexpr int kLoopIter = kLoopIterPerColumn * columns;
      // Size in bits of the loop iterator over kLoopIter iterations
      constexpr int kLoopIterBitSize =
          fpga_tools::BitsForMaxValue<kLoopIter + 1>();

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li_a = 0; li_a < kLoopIter; li_a++) {
        fpga_tools::NTuple<TT, pipe_size> pipe_read = AIn::read();

        int write_idx;
        int a_col_index;
        if constexpr (k_column_order) {
          write_idx = li_a % kLoopIterPerColumn;
          a_col_index = li_a / kLoopIterPerColumn;
        } else {
          write_idx = li_a / columns;
          a_col_index = li_a % columns;
        }
        // int write_idx = li_a / columns;

        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
            if (write_idx == k) {
              if constexpr (k * pipe_size + t < rows) {
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

      // Compute the QR Decomposition

      // Initialization of the i and j variables for the triangular loop
      int i = 0;
      int j = 0;

      constexpr int kTotalIterations = (raw_latency+1)*(columns-1) + 2;

      TT a_i_m_1[columns];

      [[intel::fpga_register]]
      TT a_i[columns];
      TT s[columns];
      TT ir;
      TT p;

      int r_index = 0;

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      [[intel::ivdep(raw_latency)]]      // NO-FORMAT: Attribute
      for (int it = 0; it < kTotalIterations; it++) {

        // PRINTF("i: %d, j: %d\n", int(i), int(j));

        if(j<columns+1){

          TT mult_lhs[rows];
          TT mult_rhs[rows];
          TT add[rows];

          TT mult_add[rows];

          TT dp{0};

          fpga_tools::UnrolledLoop<rows>([&](auto k) {

            TT a_j[rows];
            if (i <= 1) {
              a_j[k] = a_load[j].template get<k>();
            }
            else if (j < columns) {
              a_j[k] = a_compute[j].template get<k>();
            }
            else {
              a_j[k] = 0;
            }

            if (i == j) {
              a_i_m_1[k] = a_i[k];
              // if (k==0) {
              //   PRINTF("i==j a_i read\n");
              // }
            }

            if (j == columns){
                mult_lhs[k] = a_i[k];
                // if (k==0) {
                //   PRINTF("j=col a_i read\n");
                // }
                mult_rhs[k] = ir;
                add[k] = 0;
            }
            else if (i > 0) {
                mult_lhs[k] = -s[j];
                mult_rhs[k] = a_i_m_1[k];
                add[k] = a_j[k];
            }
            else {
                mult_lhs[k] = 0;
                mult_rhs[k] = 0;
                add[k] = 0;
            }

            mult_add[k] = mult_lhs[k] * mult_rhs[k] + add[k];

            if (j == columns) {
              q_a_result[i].template get<k>() = mult_add[k];
            }
            else if (i > 0) {
              a_compute[j].template get<k>() = mult_add[k];
              a_j[k] = mult_add[k];
            }

            if (j==i) {
              a_i[k] = a_j[k];
              // if (k==0) {
              //   PRINTF("j=i a_i write\n");
              // }
            }


            dp += a_i[k] * a_j[k];
          });

          if (j==i) {
            p = dp;
            ir = sycl::rsqrt(p);
            r_a_result[r_index] = sycl::sqrt(p);
            r_index++;
          }
          else if (j != columns) {
            s[j] = dp/p;
            r_a_result[r_index] = dp*ir;
            r_index++;
          }
        }

        int upper_j_bound = raw_latency + i;

        // Update loop indexes
        if (j == upper_j_bound) {
          // If i reached an index at which the j inner loop doesn't have
          // enough time to write its result for the next i iteration,
          // some "dummy" iterations are introduced
          j = i + 1;
          i = i + 1;
        } else {
          j = j + 1;
        }

      }  // end of s

      // Number of upper-right elements in the R output matrix
      constexpr int kRMatrixSize = columns * (columns + 1) / 2;

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (int r_idx = 0; r_idx < kRMatrixSize; r_idx++) {
        ROut::write(r_a_result[r_idx]);
      }

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
                  get[t] ? q_a_result[li / kLoopIterPerColumn]
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
