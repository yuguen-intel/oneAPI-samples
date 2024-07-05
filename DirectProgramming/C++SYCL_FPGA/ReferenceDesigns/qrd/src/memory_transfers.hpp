#ifndef __MEMORY_TRANSFERS_HPP__
#define __MEMORY_TRANSFERS_HPP__

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

/*
  Read matrix_count matrices of type TT from DDR by bursts of num_elem_per_bank
  elements, and write the matrices to the "MatrixPipe" pipe num_elem_per_bank by
  num_elem_per_bank elements.
  Repeat this operations "repetitions" times.
*/
template <typename TT,            // Datatype of the elements of the matrix
          int rows,               // Number of rows of the matrix
          int columns,            // Number of columns of the matrix
          int num_elem_per_bank,  // Number of TT elements per DDR burst access
          typename MatrixPipe     // Output matrix pipe
          >
void MatrixReadFromDDRToPipe(
    TT* matrix_ptr,    // Input matrix pointer
    int matrix_count,  // Number of matrix to read from DDR
    int repetitions    // Number of time to write the same matrix to the pipe
) {
#ifdef INTERLEAVED
#ifdef THREE_WAY_INTERLEAVING
  constexpr int kInterleavingFactor = 3;
#else
  constexpr int kInterleavingFactor = 2;
#endif
#else
  constexpr int kInterleavingFactor = 1;
#endif

  // We may perform an incomplete memory read if the number of elements per row
  // is not a multiple of the DDR burst size
  constexpr bool kIncompleteBurst = rows % num_elem_per_bank != 0;
  constexpr int kExtraIteration = kIncompleteBurst ? 1 : 0;
  // Number of DDR burst reads of num_elem_per_bank elements required to read a
  // full column
  constexpr int kLoopIterPerColumn = rows / num_elem_per_bank + kExtraIteration;
  // Number of DDR burst reads of num_elem_per_bank to read all the matrices
  constexpr int kLoopIter = kLoopIterPerColumn * columns;
  // Size in bits of the loop iterator over kLoopIter iterations
  constexpr int kLoopIterBitSize =
      fpga_tools::BitsForMaxValue<kLoopIter * kInterleavingFactor + 1>();
  // Size of a full matrix
  constexpr int kMatrixSize = rows * columns;

#if defined(IS_BSP)
  // When targeting a BSP, we instruct the compiler that this pointer
  // lives on the device.
  // Knowing this, the compiler won't generate hardware to
  // potentially get data from the host.
  sycl::device_ptr<TT> matrix_ptr_located(matrix_ptr);
#else
  // Device pointers are not supported when targeting an FPGA
  // family/part
  TT* matrix_ptr_located(matrix_ptr);
#endif

  // Type used to store the matrices in the compute loop
  using column_tuple = fpga_tools::NTuple<TT, rows>;

  // Break memories up to store 4 complex numbers (32 bytes) per bank
  constexpr short kBankwidth = num_elem_per_bank * sizeof(TT);
  constexpr unsigned short kNumBanks = rows / num_elem_per_bank;

  // When specifying numbanks for a memory, it must be a power of 2.
  // Unused banks will be automatically optimized away.
  constexpr short kNumBanksNextPow2 =
      fpga_tools::Pow2(fpga_tools::CeilLog2(kNumBanks));

  [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
  [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
  [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
  [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
  column_tuple a_load[columns * kInterleavingFactor];

  // Repeatedly read matrix_count matrices from DDR and sends them to the
  // pipe
  for (int repetition = 0; repetition < repetitions; repetition++) {
    for (short matrix_index = 0; matrix_index < matrix_count;
         matrix_index += kInterleavingFactor) {
      // Keep track of the current element index in the matrix
      // Only useful in the case of kIncompleteBurst
      int load_index = 0;

      // [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0;
           li < kLoopIter * kInterleavingFactor; li++) {
        bool last_burst_of_col;
        if constexpr (kIncompleteBurst) {
          // Check if we are reading the last DDR burst of the current column
          last_burst_of_col =
              (li % kLoopIterPerColumn) == kLoopIterPerColumn - 1;
        }

        fpga_tools::NTuple<TT, num_elem_per_bank> ddr_read;

        // Perform the DDR burst read of num_elem_per_bank elements
        fpga_tools::UnrolledLoop<num_elem_per_bank>([&](auto k) {
          if constexpr (kIncompleteBurst) {
            // Check if the current read index is beyond the end of the current
            // matrix column
            bool out_of_bounds =
                last_burst_of_col &&
                ((k % num_elem_per_bank) > ((rows - 1) % num_elem_per_bank));

            // Only perform the DDR reads that are relevant (and don't access a
            // memory address that may be beyond the matrix last address)
            if (!out_of_bounds) {
              ddr_read.template get<k>() =
                  matrix_ptr_located[matrix_index * kMatrixSize + load_index +
                                     k];
            }
          } else {
            ddr_read.template get<k>() =
                matrix_ptr_located[matrix_index * kMatrixSize +
                                   (int)(li)*num_elem_per_bank + k];
          }
        });

        if constexpr (kIncompleteBurst) {
          // Update the current element index in the input matrix according
          // to the read size of the current iteration
          load_index +=
              last_burst_of_col ? rows % num_elem_per_bank : num_elem_per_bank;
        }

        bool second_matrix = li >= kLoopIter;
        int aj_offset = second_matrix ? kLoopIter : 0;
        int acol_offset = second_matrix ? columns : 0;
#ifdef THREE_WAY_INTERLEAVING
        bool third_matrix = li >= kLoopIter*2;
        if (third_matrix) {
          aj_offset = kLoopIter*2;
          acol_offset = 2*columns;
        }
#endif



        int aj_li = int(li) - aj_offset;

        int write_idx = aj_li % kLoopIterPerColumn;
        int a_col_index = aj_li / kLoopIterPerColumn + acol_offset;

        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          fpga_tools::UnrolledLoop<num_elem_per_bank>([&](auto t) {
            if (write_idx == k) {
              constexpr int get_value = k * num_elem_per_bank + t;

              if constexpr (get_value < rows) {
                a_load[a_col_index].template get<get_value>() =
                    ddr_read.template get<t>();
              }
            }
            // Delay data signals to create a vine-based data distribution
            // to lower signal fanout.
            ddr_read.template get<t>() =
                sycl::ext::intel::fpga_reg(ddr_read.template get<t>());
          });

          write_idx = sycl::ext::intel::fpga_reg(write_idx);
        });

      }  // end of li

      for (int times = 0; times < 2; times++) {
        for (int col = 0; col < columns*kInterleavingFactor; col++) {
          MatrixPipe::write(a_load[col]);
        }
      }

    }  // end of matrix_index
  }    // end of repetition
}

/*
  Write matrix_count matrices of type TT from a pipe, num_elem_per_bank by
  num_elem_per_bank and write them to DDR by bursts of num_elem_per_bank
  elements.
  Repeat this operations "repetitions" times.
*/
template <typename TT,            // Datatype of the elements of the matrix
          int rows,               // Number of rows of the matrix
          int columns,            // Number of columns of the matrix
          int num_elem_per_bank,  // Number of TT elements per DDR burst access
          typename MatrixPipe     // Input matrix
          >
void MatrixReadPipeToDDR(
    TT* matrix_ptr,    // Output matrix pointer
    int matrix_count,  // Number of matrix to write to DDR
    int repetitions    // Number of time to read the same matrix to the pipe
) {
#ifdef INTERLEAVED
#ifdef THREE_WAY_INTERLEAVING
  constexpr int kInterleavingFactor = 3;
#else
  constexpr int kInterleavingFactor = 2;
#endif
#else
  constexpr int kInterleavingFactor = 1;
#endif

  // We may perform an incomplete memory write if the number of elements per row
  // is not a multiple of the DDR burst size
  constexpr bool kIncompleteBurst = rows % num_elem_per_bank != 0;
  constexpr int kExtraIteration = kIncompleteBurst ? 1 : 0;
  // Number of DDR burst of num_elem_per_bank required to write a full column
  constexpr int kLoopIterPerColumn = rows / num_elem_per_bank + kExtraIteration;
  // Number of DDR burst of num_elem_per_bank to write all the matrices
  constexpr int kLoopIter = kLoopIterPerColumn * columns;
  // Size in bits of the loop iterator over kLoopIter iterations
  constexpr int kLoopIterBitSize =
      fpga_tools::BitsForMaxValue<kLoopIter * kInterleavingFactor + 1>();
  // Size of a full matrix
  constexpr int kMatrixSize = rows * columns;

#if defined(IS_BSP)
  // When targeting a BSP, we instruct the compiler that this pointer
  // lives on the device.
  // Knowing this, the compiler won't generate hardware to
  // potentially get data from the host.
  sycl::device_ptr<TT> matrix_ptr_located(matrix_ptr);
#else
  // Device pointers are not supported when targeting an FPGA
  // family/part
  TT* matrix_ptr_located(matrix_ptr);
#endif

  // Type used to store the matrices in the compute loop
  using column_tuple = fpga_tools::NTuple<TT, rows>;

  // Break memories up to store 4 complex numbers (32 bytes) per bank
  constexpr short kBankwidth = num_elem_per_bank * sizeof(TT);
  constexpr unsigned short kNumBanks = rows / num_elem_per_bank;

  // When specifying numbanks for a memory, it must be a power of 2.
  // Unused banks will be automatically optimized away.
  constexpr short kNumBanksNextPow2 =
      fpga_tools::Pow2(fpga_tools::CeilLog2(kNumBanks));

  [[intel::numbanks(kNumBanksNextPow2)]]  // NO-FORMAT: Attribute
  [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
  [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
  [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
  column_tuple q_result[columns * kInterleavingFactor];

  // Repeatedly read matrix_count matrices from the pipe and write them to DDR
  for (int repetition = 0; repetition < repetitions; repetition++) {
    for (short matrix_index = 0; matrix_index < matrix_count;
         matrix_index += kInterleavingFactor) {
      
      for (int col = 0; col < columns * kInterleavingFactor; col++) {
#ifdef INTERLEAVED
        int base_addr = col / kInterleavingFactor;
#ifdef THREE_WAY_INTERLEAVING
        int offset;
        if (col%3 == 0) {
          offset = 0;
        } 
        else if (col%3 == 1) {
          offset = columns;
        }
        else {
          offset = columns*2;
        }
#else
        int offset = col%2 == 0 ? 0 : columns;
#endif
        q_result[base_addr + offset] = MatrixPipe::read();
#else
        q_result[col] = MatrixPipe::read();
#endif
      }

      // Keep track of the current element index in the output matrix
      // Only useful in the case of kIncompleteBurst
      int write_idx = 0;

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      [[intel::ivdep]]                   // NO-FORMAT: Attribute
      for (ac_int<kLoopIterBitSize, false> li = 0;
           li < kLoopIter * kInterleavingFactor; li++) {
        bool second_matrix = li >= kLoopIter;
        int aj_offset = second_matrix ? kLoopIter : 0;
        int acol_offset = second_matrix ? columns : 0;
#ifdef THREE_WAY_INTERLEAVING
        bool third_matrix = li >= kLoopIter*2;
        if (third_matrix) {
          aj_offset = kLoopIter*2;
          acol_offset = 2*columns;
        }
#endif
        int aj_li = int(li) - aj_offset;

        int column_iter = aj_li % kLoopIterPerColumn;
        bool get[kLoopIterPerColumn];
        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto k) {
          get[k] = column_iter == k;
          column_iter = sycl::ext::intel::fpga_reg(column_iter);
        });

        int a_col_index = aj_li / kLoopIterPerColumn + acol_offset;

        fpga_tools::NTuple<TT, num_elem_per_bank> pipe_write;
        fpga_tools::UnrolledLoop<kLoopIterPerColumn>([&](auto t) {
          fpga_tools::UnrolledLoop<num_elem_per_bank>([&](auto k) {
            if constexpr (t * num_elem_per_bank + k < rows) {
              auto q = q_result[a_col_index]
                           .template get<t * num_elem_per_bank + k>();
              if (get[t]) {
                pipe_write.template get<k>() = q;
              } else {
                pipe_write.template get<k>() =
                    sycl::ext::intel::fpga_reg(pipe_write.template get<k>());
              }
            }
          });
        });

        bool last_burst_of_col;
        if constexpr (kIncompleteBurst) {
          // Check if we are writing the last DDR burst of the current column
          last_burst_of_col =
              (aj_li % kLoopIterPerColumn) == kLoopIterPerColumn - 1;
        }

#ifdef INTERLEAVED
        int offset = second_matrix ? 1 : 0;
#ifdef THREE_WAY_INTERLEAVING
        if (third_matrix) {
          offset = 2;
        }
#endif
#else
        int offset = 0;
#endif

        fpga_tools::UnrolledLoop<num_elem_per_bank>([&](auto k) {
          if constexpr (kIncompleteBurst) {
            // Check if the current write index is beyond the end of the current
            // matrix column
            bool out_of_bounds =
                last_burst_of_col && (k > ((rows - 1) % num_elem_per_bank));

            // Only perform the DDR writes that are relevant (and don't access a
            // memory address that may be beyond the buffer last address)
            if (!out_of_bounds) {
              matrix_ptr_located[(matrix_index + offset) * kMatrixSize + write_idx + k] =
                  pipe_write.template get<k>();
            }
          } else {
            matrix_ptr_located[(matrix_index + offset) * kMatrixSize +
                               int(aj_li) * num_elem_per_bank + k] =
                pipe_write.template get<k>();
          }
        });

        if constexpr (kIncompleteBurst) {
          // Update the current element index in the write buffer according
          // to the write size of the current iteration
          write_idx +=
              last_burst_of_col ? rows % num_elem_per_bank : num_elem_per_bank;
        }
      }  // end of li
    }    // end of matrix_index
  }      // end of repetition
}

#endif /* __MEMORY_TRANSFERS_HPP__ */