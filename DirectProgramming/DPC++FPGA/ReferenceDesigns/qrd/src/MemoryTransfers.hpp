#pragma once

#include "Utils.hpp"
#include "QRInversionDim.hpp"

template <typename kernelName,      // Name to use for the Kernel
          bool isComplex,           // Helps identify the correct bank size
          typename TT,              // The datatype for the computation
          int rows,                 // Number of rows in the incoming A matrix
          int columns,              // Number of columns in the incoming A
                                    // matrix, must be <= kNumRows
          typename AMatrixOutPipe,  // A matrix input, receive a full column
                                    // of complex numbers with each read,
                                    // wrapped in NTuple
          short numBuffers          // number of buffers to rotate with
          >
sycl::event DDRToLocalMemoryCopy( sycl::queue& q, 
                                  size_t matrices, 
                                  sycl::buffer<TT, 1> * A_buffer[numBuffers],
                                  size_t bufferIdx) {

  using dim = QRInversionDim<isComplex, rows, columns>;

  constexpr int kAMatrixSize = dim::AMatrixSize;
  constexpr int kNumElementsPerBank = dim::NumElementsPerBank;
  constexpr int kBankWidth = dim::BankWidth;
  constexpr int kNumBanks = dim::NumBanks;
  constexpr int kNumBanksNextPow2 = dim::NumBanksNextPow2;
  constexpr bool kNonCompleteIter = dim::NonCompleteIter;
  constexpr int kExtraIter = dim::ExtraIter;
  constexpr int kLoadIter = dim::LoadIter;
  constexpr int kLoadIterBitSize = dim::LoadIterBitSize;
  constexpr int kLoadItersPerColumn = dim::LoadItersPerColumn; 

  using PipeType = NTuple<TT, kNumElementsPerBank>;

  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffers
    sycl::accessor A_matrix_accessor(*A_buffer[bufferIdx], h, sycl::read_only);

    sycl::stream out(64000, 64000, h);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {

      // Go over the matrices
      for (int matrixIdx = 0; matrixIdx < matrices; matrixIdx++) {

        /*
          ================================================================
          Copy data from DDR memory to on-chip memory.
          ================================================================
        */
        // Get the index of the first bank of the current matrix l
        int loadBankIndex = matrixIdx * kAMatrixSize;

        [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
        for (ac_int<kLoadIterBitSize, false> li = 0; li < kLoadIter; 
                                                                  li++) {
          PipeType pipeData;

          bool lastRow = false;

          if constexpr(kNonCompleteIter){
            lastRow = (li%kLoadItersPerColumn) == kLoadItersPerColumn - 1;
          }

          UnrolledLoop<kNumElementsPerBank>([&](auto k) {

            bool outOfBounds = false;
            if constexpr(kNonCompleteIter){
             outOfBounds = lastRow && 
           ((k % kNumElementsPerBank) > ((rows-1) % kNumElementsPerBank));
            }

            if(!outOfBounds){
              pipeData.template get<k>() = A_matrix_accessor[loadBankIndex + k];
            }
          });

          if constexpr(kNonCompleteIter){
            int readElements = (rows % kNumElementsPerBank != 0) 
                                          && lastRow ?
                                          rows % kNumElementsPerBank :  
                                          kNumElementsPerBank;

            // Increase the bank index
            loadBankIndex += readElements;
          }
          else{
            loadBankIndex += kNumElementsPerBank;
          }

          AMatrixOutPipe::write(pipeData);
        } // end of li

      } // end of matrixIdx
    }); // end of h
  }); // end of q submit

  return e;

}

template <typename kernelName,    // Name to use for the Kernel
          bool isComplex,         // Helps identify the correct bank size
          typename TT,            // The datatype for the computation
          int rows,               // Number of rows in the incoming A matrix
          int columns,            // Number of columns in the incoming A
                                  // matrix, must be <= rows
          typename QMatrixInPipe, // Q matrix input pipe from the compute kernel
          typename RMatrixInPipe, // R matrix input pipe from the compute kernel
          short numBuffers        // number of buffers to rotate with
          >
sycl::event LocalMemoryToDDRCopy( sycl::queue& q, 
                                  size_t matrices, 
                                  sycl::buffer<TT, 1> * QR_buffer[numBuffers],
                                  size_t bufferIdx) {

  using dim = QRInversionDim<isComplex, rows, columns>;

  constexpr int kNumElementsPerBank = dim::NumElementsPerBank;
  constexpr int kBankWidth = dim::BankWidth;
  constexpr int kNumBanks = dim::NumBanks;
  constexpr int kRMatrixSize = dim::RMatrixSize;
  constexpr bool kNonCompleteIter = dim::NonCompleteIter;
  constexpr int kExtraIter = dim::ExtraIter;
  constexpr int kStoreIter = dim::StoreIter;
  constexpr int kStoreIterBitSize = dim::StoreIterBitSize;
  constexpr bool kNonCompleteBank = dim::NonCompleteBank;
  constexpr int kExtraBank = dim::ExtraBank;
  constexpr int kNumRBanks = dim::NumRBanks;

  using PipeType = NTuple<TT, kNumElementsPerBank>;

  auto e = q.submit([&](sycl::handler &h) {

    // Create accessor to the FPGA DDR buffers
    sycl::accessor QR_matrix_accessor(*QR_buffer[bufferIdx], h, sycl::write_only, 
                                                                sycl::no_init);

    sycl::stream out(64000, 64000, h);

    h.single_task<kernelName>([=]() [[intel::kernel_args_restrict]] {
      

      // Go over the matrices
      for (int matrixIdx = 0; matrixIdx < matrices; matrixIdx++) {

        /*
          ================================================================
          Copy the result from on-chip memory to DDR memory.
          ================================================================
        */
        int qr_idx = 0;
        [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
        for (int si = 0; si < kNumRBanks; si++) {
          PipeType pipeData = RMatrixInPipe::read();

          UnrolledLoop<kNumElementsPerBank>([&](auto k) {
            if ((qr_idx + k) < kRMatrixSize){
              QR_matrix_accessor[qr_idx + k] = pipeData.template get<k>();
            }
          });

          qr_idx += kNumElementsPerBank;
        }

        [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
        for (ac_int<kStoreIterBitSize, false> si = 0; si < kStoreIter; 
                                                                  si++) {
          PipeType pipeData = QMatrixInPipe::read();
#define WORKING
#ifdef WORKING
          TT pipeDataArray[kNumElementsPerBank];
          UnrolledLoop<kNumElementsPerBank>([&](auto k) {
            pipeDataArray[k] = pipeData.template get<k>();
          });
#endif

          bool lastRow = false;
          if constexpr(kNonCompleteIter){
            lastRow = si % kNumBanks == kNumBanks-1; 
          } 

#ifdef WORKING
          #pragma unroll 
          for(int k = 0; k<kNumElementsPerBank; k++){
            bool outOfBounds = false;
            if constexpr(kNonCompleteIter){
              outOfBounds = lastRow && 
                    (k > ((rows-1) % kNumElementsPerBank));
            }

            if(!outOfBounds){
              QR_matrix_accessor[qr_idx + k] = pipeDataArray[k];
            }
          }
#else
          // Finally, the kNumElementsPerBank elements from bank are 
          // written to the QR_matrix_accessor
          UnrolledLoop<kNumElementsPerBank>([&](auto k) {
            bool outOfBounds = false;
            if constexpr(kNonCompleteIter){
              outOfBounds = lastRow && 
                    (k > ((rows-1) % kNumElementsPerBank));
            }

            if(!outOfBounds){
              QR_matrix_accessor[qr_idx + k] = pipeData.template get<k>();
            }
          });
#endif

          if constexpr(kNonCompleteIter){
            int wroteElements = lastRow ? rows % kNumElementsPerBank :  
                                                      kNumElementsPerBank;
            qr_idx += wroteElements;
          }
          else{
            qr_idx += kNumElementsPerBank;
          }                
        } // end for si=0:kStoreIter-1
      } // end for matrixIdx=0:matrices-1
    }); // end of single task
  }); // end of q submit

  return e;

}