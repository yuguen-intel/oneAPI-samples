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
          typename AIn,     // A matrix input pipe, receive pipe_size
                            // elements from the pipe with each read
          typename QOut,    // Q matrix output pipe, send pipe_size
                            // elements to the pipe with each write
          typename ROut>
struct StreamingQRD {
  // Set the computation type to T or ac_complex<T> depending on the value
  // of is_complex
  using TT = std::conditional_t<is_complex, ac_complex<T>, T>;

#if defined(IS_BSP)

#else
  // kernel property method to config invocation interface
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::
            streaming_interface_remove_downstream_stall};
  }
#endif

  void operator()() const {
    // Functional limitations
    static_assert(rows >= columns,
                  "only rectangular matrices with rows>=columns are supported");
    static_assert(columns >= 4,
                  "only matrices of size 4x4 and over are supported");

    // Type used to store the matrices in the compute loop
    using column_tuple = fpga_tools::NTuple<TT, rows>;

    constexpr int kNumElementsPerBank = is_complex ? 4 : 8;
    // Break memories up to store 4 complex numbers (32 bytes) per bank
    constexpr short kBankwidth = kNumElementsPerBank * sizeof(TT);
    constexpr unsigned short kNumBanks = rows / kNumElementsPerBank;

    // When specifying numbanks for a memory, it must be a power of 2.
    // Unused banks will be automatically optimized away.
    constexpr short kNumBanksNextPow2 =
        fpga_tools::Pow2(fpga_tools::CeilLog2(kNumBanks));

    constexpr int kAlwaysExtraIterations = raw_latency > columns;
    constexpr int kIterationCountAlwaysExtraIterations =
        (raw_latency + 1) * columns;
    constexpr int kIterationCountNotAlwaysExtraIterations =
        columns * (columns + 1) / 2 + columns +
        raw_latency * (raw_latency - 1) / 2;
    constexpr int kTotalIterations =
        kAlwaysExtraIterations ? kIterationCountAlwaysExtraIterations
                               : kIterationCountNotAlwaysExtraIterations;

    constexpr int kUpperBound = raw_latency;

    while (1) {
      // Compute the QR Decomposition

      [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      column_tuple a_compute[columns];

      // Initialization of the i and j variables for the triangular loop
      int i = 0;
      int j = 0;
      int j_count = kAlwaysExtraIterations ? 1 : raw_latency - columns + 1;

      TT a_i_m_1[columns];
      TT a_i[columns];

      [[intel::fpga_memory]] TT s_or_ir[columns];

      T p, ir;


      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      [[intel::ivdep(raw_latency)]]      // NO-FORMAT: Attribute
      for (int it = 0; it < kTotalIterations; it++) {
        // PRINTF("i: %d, j: %d\n", int(i), int(j));

        TT dp{0};

        column_tuple pipe_read, pipe_write;
        if ((i <= 1) && (j < columns)) {
// #if defined(IS_BSP)
          pipe_read = AIn::read();

// #else
//           bool oka;
//           pipe_read = AIn::read(oka);
// #endif
          // fpga_tools::UnrolledLoop<rows>([&](auto k) {
          //   pipe_read.template get<k>() = AIn::template PipeAt<k>::read();
          // });
        }

        fpga_tools::UnrolledLoop<rows>([&](auto k) {
          TT a_j;
          if ((i <= 1) && (j < columns)) {
            a_j = pipe_read.template get<k>();
            // a_j = a_in[j * columns + k];
            // a_j = a_load[j].template get<k>();
          } else if (j < columns) {
            a_j = a_compute[j].template get<k>();
          } else {
            a_j = 0;
          }

          if (j == i - 1) {
            a_i_m_1[k] = a_j;
          }

          TT mult_lhs = (i > 0) ? a_i_m_1[k] : TT{0};
          TT mult_rhs = (i > 0) && (j < columns) ? s_or_ir[j] : TT{0};
          TT add = (i > 0) && (j != i - 1) ? a_j : TT{0};

          TT mult_add = mult_lhs * mult_rhs + add;

          if (i > 0) {
            if (j == i - 1) {
              // q_a_result[i - 1].template get<k>() = mult_add;
              // q_out[(i - 1) * columns + k] = mult_add;
              // QOut::template PipeAt<k>::write(mult_add);
              pipe_write.template get<k>() = mult_add;
            } else if (j < columns) {
              a_compute[j].template get<k>() = mult_add;
              a_j = mult_add;
            }
          }

          if (j == i) {
            if constexpr (is_complex) {
              a_i[k] = a_j.conj();
            } else {
              a_i[k] = a_j;
            }
          }

          dp += a_j * a_i[k];
        });

        if (i > 0) {
          if (j == i - 1) {
// #if defined(IS_BSP)
            QOut::write(pipe_write);
// #else
//             bool okq;
//             QOut::write(pipe_write, okq);

// #endif
          }
        }

        TT s_ir_val;
        TT r_val;
        if (j == i) {
          if constexpr (is_complex) {
            p = dp.r();
          } else {
            p = dp;
          }
          ir = sycl::rsqrt(p);
          r_val = sycl::sqrt(p);

          s_ir_val = ir;
        } else if (j > i) {
          s_ir_val = -dp / p;
          r_val = dp * ir;
        }

        if ((j != i - 1) && (j < columns)) {
// #if defined(IS_BSP)
          ROut::write(r_val);
// #else
//           bool okr;
//           ROut::write(r_val, okr);
// #endif
          // r_out[r_index] = r_val;
          // r_a_result[r_index] = r_val;
          // r_index++;
          s_or_ir[j] = s_ir_val;
        }

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
    }

  }  // end of operator
};   // end of struct

}  // namespace fpga_linalg

#endif /* __STREAMING_QRD_HPP__ */
