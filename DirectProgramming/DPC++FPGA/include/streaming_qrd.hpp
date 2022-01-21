#ifndef __STREAMING_QRD_HPP__
#define __STREAMING_QRD_HPP__

#include "tuple.hpp"
#include "utils.hpp"
#include "unrolled_loop.hpp"
#include "metaprogramming_math.hpp"

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
template <typename T,        // The datatype for the computation
          bool is_complex,   // True if T is ac_complex<X>
          int rows,          // Number of rows in the incoming A matrices
          int columns,       // Number of columns in the incoming A
                             // matrices, must be <= rows
          int raw_latency,    // Read after write latency (in iterations) of
                             // the triangular loop of this function.
                             // This value depends on the FPGA target, the
                             // datatype, the target frequency, etc.
                             // This value will have to be tuned for optimal
                             // performance. Refer to the Triangular Loop
                             // design pattern tutorial.
                             // In general, find a high value for which the
                             // compiler is able to achieve an II of 1 and
                             // go down from there.
          int matrix_count,  // Number of matrices to read from the input
                             // pipe sequentially
          int pipe_size,     // Number of elements read/write per pipe
                             // operation
          typename AIn,      // A matrix input pipe, receive pipe_size
                             // elements from the pipe with each read
          typename QOut,     // Q matrix output pipe, send pipe_size
                             // elements to the pipe with each write
          typename ROut      // R matrix output pipe, send pipe_size
                             // elements to the pipe with each write.
                             // Only upper-right elements of R are
                             // sent in row order, starting with row 0.
          >
struct StreamingQRD {
  void operator()() const {
    // Functional limitations
    static_assert(rows >= columns,
                  "only rectangular matrices with rows>=columns are supported");
    static_assert((columns <= 512) && (columns >= 4),
                  "only matrices of size 4x4 to 512x512 are supported");

    /*
      This code implements a OneAPI optimized variation of the following
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
    using column_tuple = NTuple<TT, rows>;

    // Number of upper-right elements in the R output matrix
    constexpr int kRMatrixSize = columns * (columns + 1) / 2;
    // Fanout reduction factor for signals that fanout to rows compute cores
    constexpr int kFanoutReduction = 8;
    // Number of signal replication required to cover all the rows compute cores
    // given a kFanoutReduction factor
    constexpr int kBanksForFanout = (rows % kFanoutReduction)
                                    ? (rows / kFanoutReduction) + 1
                                    : rows / kFanoutReduction;

    // Number of iterations performed without any dummy work added for the
    // triangular loop optimization
    constexpr int kVariableIterations = columns - raw_latency;
    // Total number of dummy iterations
    static constexpr int kDummyIterations =
        raw_latency > columns
            ? (columns - 1) * columns / 2 + (raw_latency - columns) * columns
            : raw_latency * (raw_latency - 1) / 2;
    // Total number of iterations (including dummy iterations)
    static constexpr int kIterations =
        columns + columns * (columns + 1) / 2 + kDummyIterations;

    // Size in bits of the "i" loop variable in the triangular loop
    // i starts from -1 as we are doing a full copy of the matrix read from the
    // pipe to a "compute" matrix before starting the decomposition
    constexpr int kIBitSize = fpga_tools::BitsForMaxValue<rows + 1>() + 1;

    // j starts from i, so from -1 and goes up to columns
    // So we need:
    // -> enough bits to encode columns+1 for the positive iterations and
    //    the exit condition
    // -> one extra bit for the -1
    // But j may start below -1 if we perform more dummy iterations than the
    // number of columns in the matrix.
    // In that case, we need:
    // -> enough bits to encode columns+1 for the positive iterations and
    //    the exit condition
    // -> enough bits to encode the maximum number of negative iterations
    static constexpr int kJNegativeIterations =
        kVariableIterations < 0 ? -kVariableIterations : 1;
    static constexpr int kJBitSize = fpga_tools::BitsForMaxValue<columns + 1>()
                          + fpga_tools::BitsForMaxValue<kJNegativeIterations>();

    // Iterate over the number of matrices to decompose per function call
    for (int matrix_iter = 0; matrix_iter < matrix_count; matrix_iter++) {
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
      column_tuple a_load[columns], a_compute[columns], q_result[columns];

      // Contains the values of the upper-right part of R in a row by row
      // fashion, starting by row 0
      [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      TT r_result[kRMatrixSize];

      // Copy a matrix from the pipe to a local memory
      // Number of DDR burst reads of pipe_size required to read a full column
      constexpr int kExtraIteration = (rows % pipe_size) != 0 ? 1 : 0;
      constexpr int kLoopIterPerColumn = rows / pipe_size + kExtraIteration;
      // Number of DDR burst reads of pipe_size to read all the matrices
      constexpr int kLoopIter = kLoopIterPerColumn * columns;
      // Size in bits of the loop iterator over kLoopIter iterations
      constexpr int kLoopIterBitSize = 
                                  fpga_tools::BitsForMaxValue<kLoopIter + 1>();

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        PipeTable<pipe_size, TT> pipe_read = AIn::read();

        int write_idx = li % kLoopIterPerColumn;

        UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          UnrolledLoop<pipe_size>([&](auto t) {
            if (write_idx == k) {
              if constexpr (k * pipe_size + t < rows) {
                a_load[li / kLoopIterPerColumn]
                    .template get<k * pipe_size + t>() = pipe_read.elem[t];
              }
            }

            // Delay data signals to create a vine-based data distribution
            // to lower signal fanout.
            pipe_read.elem[t] = sycl::ext::intel::fpga_reg(pipe_read.elem[t]);
          });

          write_idx = sycl::ext::intel::fpga_reg(write_idx);
        });
      }

      // Compute the QR Decomposition

      // r_result write index
      int r_element_index = 0;

      // a local copy of a_{i+1} that is used across multiple j iterations
      // for the computation of pip1 and p
      TT a_ip1[rows];
      // a local copy of a_ip1 that is used across multiple j iterations
      // for the computation of a_j
      TT a_i[rows];
      // Depending on the context, will contain:
      // -> -s[j]: for all the iterations to compute a_j
      // -> ir: for one iteration per j iterations to compute Q_i
      [[intel::fpga_memory]] 
      [[intel::private_copies(2)]] // NO-FORMAT: Attribute
      TT s_or_ir[columns];

      T pip1, ir;

      // Initialization of the i and j variables for the triangular loop
      ac_int<kIBitSize, true> i = -1;
      ac_int<kJBitSize, true> j = 0;

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      [[intel::ivdep(raw_latency)]]       // NO-FORMAT: Attribute
      for (int s = 0; s < kIterations; s++) {
        // Pre-compute the next values of i and j
        ac_int<kIBitSize, true> next_i;
        ac_int<kJBitSize, true> next_j;
        if (j == columns - 1) {
          // If i reached an index at which the j inner loop don't have
          // enough time to write its result for the next i iteration,
          // some "dummy" iterations are introduced
          next_j = (kVariableIterations > i)
                      ? ac_int<kJBitSize, true>{i + 1}
                      : ac_int<kJBitSize, true>{kVariableIterations};
          next_i = i + 1;
        } else {
          next_j = j + 1;
          next_i = i;
        }

        // Two matrix columns for partial results.
        TT col[rows];
        TT col1[rows];

        // Current value of s_or_ir depending on the value of j
        // It is replicated kFanoutReduction times to reduce fanout
        TT s_or_ir_j[kBanksForFanout];

        // All the control signals are precomputed and replicated
        // kFanoutReduction times to reduce fanout
        bool j_eq_i[kBanksForFanout], i_gt_0[kBanksForFanout],
            i_ge_0_j_ge_i[kBanksForFanout], j_eq_i_plus_1[kBanksForFanout],
            i_lt_0[kBanksForFanout];

        UnrolledLoop<kBanksForFanout>([&](auto k) {
          i_gt_0[k] = sycl::ext::intel::fpga_reg(i > 0);
          i_lt_0[k] = sycl::ext::intel::fpga_reg(i < 0);
          j_eq_i[k] = sycl::ext::intel::fpga_reg(j == i);
          i_ge_0_j_ge_i[k] = sycl::ext::intel::fpga_reg(i >= 0 && j >= i);
          j_eq_i_plus_1[k] = sycl::ext::intel::fpga_reg(j == i + 1);
          s_or_ir_j[k] = sycl::ext::intel::fpga_reg(s_or_ir[j]);
        });

        // Preload col and a_i with the correct data for the current iteration
        // These are going to be use to compute the dot product of two
        // different columns of the input matrix.
        UnrolledLoop<rows>([&](auto k) {
          // find which fanout bank this unrolled iteration is going to use
          constexpr auto fanout_bank_idx = k / kFanoutReduction;

          // Load col with the current column of matrix A.
          // At least one iteration of the outer loop i is required
          // for the "working copy" a_compute to contain data.
          // If no i iteration elapsed, we must read the column of
          // matrix A directly from the a_load; col then contains a_j

          if (i_gt_0[fanout_bank_idx]) {
            col[k] = a_compute[j].template get<k>();
          }
          // Using an else statement makes the compiler throw an
          // inexplicable warning when using non complex types:
          // "Compiler Warning: Memory instruction with unresolved
          // pointer may lead to bad QoR."
          if (!i_gt_0[fanout_bank_idx]) {
            col[k] = a_load[j].template get<k>();
          }

          // Load a_i for reuse across j iterations
          if (j_eq_i[fanout_bank_idx]) {
            a_i[k] = col[k];
          }
        });

        UnrolledLoop<rows>([&](auto k) {
          // find which fanout bank this unrolled iteration is going to use
          constexpr auto fanout_bank_idx = k / kFanoutReduction;

          // Depending on the iteration this code will compute either:
          // -> If i=j, a column of Q: Q_i = a_i*ir
          //    In that case, no term is added to the mult_add construct
          // -> If i!=j, an updated column of a: a_j - s[j]*a_i
          //    There is a special case if i<0 where a_j is unmodified
          //    but the i iteration is still required to fill ir and s
          //    for subsequent iterations
          auto prod_lhs = a_i[k];
          auto prod_rhs = i_lt_0[fanout_bank_idx] ? 
                                          TT{0.0} : s_or_ir_j[fanout_bank_idx];
          auto add = j_eq_i[fanout_bank_idx] ? TT{0.0} : col[k];
          if constexpr (is_complex) {
            col1[k] = prod_lhs * prod_rhs.conj() + add;
          } else {
            col1[k] = prod_lhs * prod_rhs + add;
          }

          // Store Q_i in q_result and the modified a_j in a_compute
          // To reduce the amount of control, q_result and a_compute
          // are both written to for each iteration of i>=0 && j>=i
          // In fact:
          // -> q_result could only be written to at iterations i==j
          // -> a_compute could only be written to at iterations
          //    j!=i && i>=0
          // The extra writes are harmless as the locations written to
          // are either going to be:
          // -> overwritten for the matrix Q (q_result)
          // -> unused for the a_compute
          if (i_ge_0_j_ge_i[fanout_bank_idx]) {
            q_result[j].template get<k>() = col1[k];
            a_compute[j].template get<k>() = col1[k];
          }

          // Store a_{i+1} for subsequent iterations of j
          if (j_eq_i_plus_1[fanout_bank_idx]) {
            a_ip1[k] = col1[k];
          }
        });

        // Perform the dot product <a_{i+1},a_{i+1}> or <a_{i+1}, a_j>
        TT p_ij{0.0};
        UnrolledLoop<rows>([&](auto k) {
          if constexpr (is_complex) {
            p_ij = p_ij + col1[k] * a_ip1[k].conj();
          } else {
            p_ij = p_ij + col1[k] * a_ip1[k];
          }
        });

        // Compute pip1 and ir based on the results of the dot product
        if (j == i + 1) {
          if constexpr (is_complex) {
            pip1 = p_ij.r();
            ir = sycl::rsqrt(p_ij.r());
          } else {
            pip1 = p_ij;
            ir = sycl::rsqrt(p_ij);
          }
        }

        // Compute the value of -s[j]
        TT s_j;
        if constexpr (is_complex) {
          s_j = TT{0.0f - (p_ij.r()) / pip1, p_ij.i() / pip1};
        } else {
          s_j = -p_ij / pip1;
        }

        // j may be negative if the number of "dummy" iterations is
        // larger than the matrix size
        if (j >= 0) {
          if constexpr (is_complex) {
            s_or_ir[j] =
                TT{j == i + 1 ? ir : s_j.r(), j == i + 1 ? 0.0f : s_j.i()};
          } else {
            s_or_ir[j] = j == i + 1 ? ir : s_j;
          }
        }

        // Compute the R_{i+1,i+1} or R_{i+1,j}
        TT r_ip1j;
        if constexpr (is_complex) {
          r_ip1j = j == i + 1 ? TT{sycl::sqrt(pip1), 0.0}
                              : TT{ir * p_ij.r(), ir * p_ij.i()};
        } else {
          r_ip1j = j == i + 1 ? sycl::sqrt(pip1) : ir * p_ij;
        }

        // Write the computed R value when j is not a "dummy" iteration
        if ((j >= i + 1) && (i + 1 < columns)) {
          r_result[r_element_index] = r_ip1j;
          r_element_index++;
        }

        // Update loop indexes
        j = next_j;
        i = next_i;

      }  // end of s

      // Number of upper-right elements in the R output matrix
      constexpr int kRMatrixSize = columns * (columns + 1) / 2;

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (int r_idx = 0; r_idx < kRMatrixSize; r_idx++) {
        ROut::write(r_result[r_idx]);
      }

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0; li < kLoopIter; li++) {
        int column_iter = li % kLoopIterPerColumn;
        bool get[kLoopIterPerColumn];
        UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          get[k] = column_iter == k;
          column_iter = sycl::ext::intel::fpga_reg(column_iter);
        });

        PipeTable<pipe_size, TT> pipe_write;
        UnrolledLoop<kLoopIterPerColumn>([&](auto t) {
          UnrolledLoop<pipe_size>([&](auto k) {
            if constexpr (t * pipe_size + k < rows) {
              pipe_write.elem[k] =
                  get[t] ? q_result[li / kLoopIterPerColumn]
                               .template get<t * pipe_size + k>()
                         : sycl::ext::intel::fpga_reg(pipe_write.elem[k]);
            }
          });
        });
        QOut::write(pipe_write);
      }

    }  // end of matrix_iter
  }    // end of operator
};     // end of struct

#endif /* __STREAMING_QRD_HPP__ */