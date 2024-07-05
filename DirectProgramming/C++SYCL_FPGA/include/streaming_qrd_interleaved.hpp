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
          typename AIn,     // A matrix input pipe, receive pipe_size
                            // elements from the pipe with each read
          typename QOut,    // Q matrix output pipe, send pipe_size
                            // elements to the pipe with each write
          typename ROut     // R matrix output pipe, send pipe_size
                            // elements to the pipe with each write.
                            // Only upper-right elements of R are
                            // sent in row order, starting with row 0.
          >
struct StreamingQRD {
  void operator()() const {
    // Functional limitations
    static_assert(rows >= columns,
                  "only rectangular matrices with rows>=columns are supported");
    static_assert(columns >= 4,
                  "only matrices of size 4x4 and over are supported");


#ifdef THREE_WAY_INTERLEAVING
    static_assert(raw_latency >= 3 * columns,
                  "raw_latency must be at least two times the size of the "
                  "matrix for interleaving to work");

    constexpr int kInterleavingFactor = 3;
#else
    static_assert(raw_latency >= 2 * columns,
                  "raw_latency must be at least two times the size of the "
                  "matrix for interleaving to work");

    constexpr int kInterleavingFactor = 2;
#endif

    // Set the computation type to T or ac_complex<T> depending on the value
    // of is_complex
    using TT = std::conditional_t<is_complex, ac_complex<T>, T>;

    // Type used to store the matrices in the compute loop
    using column_tuple = fpga_tools::NTuple<TT, rows>;

    // Break memories up to store 4 complex numbers (32 bytes) per bank
    constexpr int kNumElementsPerBank = is_complex ? 4 : 8;
    constexpr short kBankwidth = kNumElementsPerBank * sizeof(TT);
    constexpr unsigned short kNumBanks = rows / kNumElementsPerBank;

    // When specifying numbanks for a memory, it must be a power of 2.
    // Unused banks will be automatically optimized away.
    constexpr short kNumBanksNextPow2 =
        fpga_tools::Pow2(fpga_tools::CeilLog2(kNumBanks));

    // Compute QRDs as long as matrices are given as inputs
    while (1) {
      [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      column_tuple a_compute[columns * kInterleavingFactor];

      // Compute the QR Decomposition
      constexpr int kExtraIterationsForInterleaving = columns*(kInterleavingFactor-1);
      constexpr int kAlwaysExtraIterations = raw_latency > columns;
      constexpr int kIterationCountAlwaysExtraIterations =
          (raw_latency + 1) * columns + kExtraIterationsForInterleaving;
      constexpr int kIterationCountNotAlwaysExtraIterations =
          columns * (columns + 1) / 2 + columns +
          raw_latency * (raw_latency - 1) / 2;
      constexpr int kTotalIterations =
          kAlwaysExtraIterations ? kIterationCountAlwaysExtraIterations
                                 : kIterationCountNotAlwaysExtraIterations;

      constexpr int kUpperBound = raw_latency;

      // Initialization of the i and j variables for the triangular loop
      int i = 0;
      int j = 0;
      int j_count = kAlwaysExtraIterations ? 1 : raw_latency - columns + 1;

      
      TT m_i[columns];
      TT m_i_m_1[columns];

      [[intel::fpga_memory]] TT s_or_ir[columns * kInterleavingFactor];

      T p; 
      T ir[kInterleavingFactor];

      constexpr int kBitsForOneMatrixAddresses =
          fpga_tools::BitsForMaxValue<columns>();
      constexpr int kBitsMatrixMultiplexing =
          fpga_tools::CeilLog2(kInterleavingFactor);

      // PRINTF("total iterations: %d\n", kTotalIterations);
      // int bubble = 0;

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      [[intel::ivdep(raw_latency)]]      // NO-FORMAT: Attribute
      for (int it = 0; it < kTotalIterations; it++) {
        // PRINTF("i: %d, j: %d\n", int(i), int(j));

        int lb = i - 1 < 0 ? 0 : i - 1;
        bool compute_a = lb <= j && j < columns;
        bool compute_b = lb + columns <= j && j < 2 * columns;

#ifdef THREE_WAY_INTERLEAVING
        bool compute_c = lb + 2 * columns <= j && j < 3 * columns;
        bool compute = compute_a || compute_b || compute_c;
        ac_int<2, false> bit_offset;
        if (compute_a) {
          bit_offset = 0;
        }
        else if (compute_b) {
          bit_offset = 1;
        }
        else {
          // compute c or no compute
          bit_offset = 2;
        }
#else
        bool compute = compute_a || compute_b;
        ac_int<1, false> bit_offset = compute_b ? 1 : 0;
#endif



        if (compute) {
          int adjusted_j = j;
          if (compute_b) {
            adjusted_j = j - columns;
          }
#ifdef THREE_WAY_INTERLEAVING
          else if (compute_c) {
            adjusted_j = j - 2 * columns;
          }
#endif

          ac_int<kBitsForOneMatrixAddresses + kBitsMatrixMultiplexing, false> aj_with_offset{
              adjusted_j};
#ifdef THREE_WAY_INTERLEAVING
          aj_with_offset[fpga_tools::CeilLog2(columns)] = bit_offset[0];
          aj_with_offset[fpga_tools::CeilLog2(columns)+1] = bit_offset[1];
#else
          aj_with_offset[fpga_tools::CeilLog2(columns)] = bit_offset;
#endif
          
          TT dp{0};

          column_tuple pipe_read, pipe_write;
          if (i <= 1) {
            pipe_read = AIn::read();
          }

          fpga_tools::UnrolledLoop<rows>([&](auto k) {
            TT m_j_k;
            if (i <= 1) {
              // m_j_k = a_load[aj_with_offset].template get<k>();
              m_j_k = pipe_read.template get<k>();
            } else {
              m_j_k = a_compute[aj_with_offset].template get<k>();
            }

            if (adjusted_j == i - 1) {
              m_i_m_1[k] = m_j_k;
            }

            TT mult_lhs = (i > 0) ? m_i_m_1[k] : TT{0};
            TT mult_rhs = (i > 0) ? s_or_ir[aj_with_offset] : TT{0};
            TT add = (i > 0) && (adjusted_j != i - 1) ? m_j_k : TT{0};

            TT mult_add = mult_lhs * mult_rhs + add;

            if (i > 0) {
              if (adjusted_j == i - 1) {
                // q_a_result[(i - 1) + offset].template get<k>() = mult_add;
                pipe_write.template get<k>() = mult_add;

              } else {
                a_compute[aj_with_offset].template get<k>() = mult_add;
                m_j_k = mult_add;
              }
            }

            if (adjusted_j == i) {
              if constexpr (is_complex) {
                m_i[k] = m_j_k.conj();
              } else {
                m_i[k] = m_j_k;
              }
            }

            dp += m_j_k * m_i[k];
          });

          if (i > 0) {
            if (adjusted_j == i - 1) {
              QOut::write(pipe_write);
            }
          }

          TT s_ir_val;
          TT r_val;
          if (adjusted_j == i) {
            if constexpr (is_complex) {
              p = dp.r();
            } else {
              p = dp;
            }

            ir[bit_offset] = sycl::rsqrt(p);
            s_ir_val = ir[bit_offset];
            r_val = sycl::sqrt(p);

          } else if (adjusted_j > i) {
            s_ir_val = -dp / p;
            r_val = dp * ir[bit_offset];
          }

          if (adjusted_j != i - 1) {

            ROut::write(r_val);

            s_or_ir[aj_with_offset] = s_ir_val;
          }
        }
        // else {
        //   bubble++;
        // }

        // Update loop indexes
        if (j_count == kUpperBound) {
          j = i;
          if constexpr (kAlwaysExtraIterations) {
            j_count = 0;
          } else {
            j_count = (columns - i - raw_latency) > 0
                          ? i + raw_latency - columns + 1
                          : 0;
          }
          i++;
        } else {
          j++;
          j_count++;
        }

      }  // end of s
      // PRINTF("bubbles: %d\n", bubble);

    }  // end of while(1)
  }    // end of operator
};     // end of struct

}  // namespace fpga_linalg

#endif /* __STREAMING_QRD_HPP__ */
