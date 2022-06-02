#include <math.h>

#include <CL/sycl.hpp>
#include <list>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"
#include "matmul.hpp"
#define DEBUG 1
/*
  COMPLEX, COLS_COMPONENT and ROWS_COMPONENT are defined by the build system.
  Depending on the value of COMPLEX, the real or complex MatrixMultiplication is
  defined

  Function arguments:
  - a_matrix:    The input matrix A.
  - b_matrix:    The input matrix B. Interpreted as a transposed matrix.
  - mm_matrix:   The MM matrix, product of A and B. The function will overwrite
  this matrix.
  - q:           The device queue.
  - matrix_count: Number of matrices to decompose.
  - repetitions: The number of repetitions of the computation to execute.
                 (for performance evaluation)
*/
#if COMPLEX == 0
// Real single precision floating-point QR Decomposition
void MatrixMultiplication(std::vector<float> &a_matrix,
                          std::vector<float> &b_matrix,
                          std::vector<float> &mm_matrix, sycl::queue &q,
                          int matrix_count, int repetitions) {
  constexpr bool is_complex = false;
  MATMULImpl<COLS_COMPONENT, ROWS_COMPONENT, is_complex, float>(
      a_matrix, b_matrix, mm_matrix, q, matrix_count, repetitions);
}
#else
// Complex single precision floating-point QR Decomposition
void MatrixMultiplication(std::vector<ac_complex<float> > &a_matrix,
                          std::vector<ac_complex<float> > &b_matrix,
                          std::vector<ac_complex<float> > &mm_matrix,
                          sycl::queue &q, int matrix_count, int repetitions) {
  constexpr bool is_complex = true;
  MATMULImpl<COLS_COMPONENT, ROWS_COMPONENT, is_complex, float>(
      a_matrix, b_matrix, mm_matrix, q, matrix_count, repetitions);
}
#endif

/*
  returns if both the real and complex parts of the given ac_complex
  value are finite
*/
bool IsFinite(ac_complex<float> val) {
  return std::isfinite(val.r()) && std::isfinite(val.i());
}

/*
  returns if the given value is finite
*/
bool IsFinite(float val) { return std::isfinite(val); }

int main(int argc, char *argv[]) {
  constexpr size_t kRandomSeed = 1138;
  constexpr size_t kRandomMin = 1;
  constexpr size_t kRandomMax = 10;
  constexpr size_t kRows = ROWS_COMPONENT;
  constexpr size_t kColumns = COLS_COMPONENT;
  constexpr size_t kMatrixSize = kRows * kColumns;
  constexpr bool kComplex = COMPLEX != 0;

  // Get the number of times we want to repeat the decomposition
  // from the command line.
#if defined(FPGA_EMULATOR)
  int repetitions = argc > 1 ? atoi(argv[1]) : 16;
#else
  int repetitions = argc > 1 ? atoi(argv[1]) : 819200;
#endif
  if (repetitions < 1) {
    std::cout << "Number of repetitions given is lower that 1." << std::endl;
    std::cout << "The decomposition must occur at least 1 time." << std::endl;
    std::cout << "Increase the number of repetitions (e.g. 16)." << std::endl;
    return 1;
  }

  constexpr size_t kMatricesToMultiply = 1;

  try {
    // SYCL boilerplate
#if defined(FPGA_EMULATOR)
    sycl::ext::intel::fpga_emulator_selector device_selector;
#else
    sycl::ext::intel::fpga_selector device_selector;
#endif

    // Enable the queue profiling to time the execution
    sycl::property_list queue_properties{
        sycl::property::queue::enable_profiling()};
    sycl::queue q = sycl::queue(device_selector, dpc_common::exception_handler,
                                queue_properties);

    sycl::device device = q.get_device();
    std::cout << "Device name: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // Select a type for this compile depending on the value of COMPLEX
    using T = std::conditional_t<kComplex, ac_complex<float>, float>;

    // Create vectors to hold all the input and output matrices
    std::vector<T> a_matrix;
    std::vector<T> b_matrix;
    std::vector<T> mm_matrix;

    a_matrix.resize(kMatrixSize * kMatricesToMultiply);
    b_matrix.resize(kMatrixSize * kMatricesToMultiply);
    mm_matrix.resize(kMatrixSize * kMatricesToMultiply);

    std::cout << "Generating " << kMatricesToMultiply << " random ";
    if constexpr (kComplex) {
      std::cout << "complex ";
    } else {
      std::cout << "real ";
    }
    std::cout << "matri" << (kMatricesToMultiply > 1 ? "ces" : "x")
              << " of size " << kRows << "x" << kColumns << " " << std::endl;

    // Generate the random input matrices
    srand(kRandomSeed);

    for (int matrix_index = 0; matrix_index < kMatricesToMultiply;
         matrix_index++) {
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          float random_real_a = rand() % (kRandomMax - kRandomMin) + kRandomMin;
#if COMPLEX == 0
          a_matrix[matrix_index * kMatrixSize + row * kColumns + col] =
              random_real_a;
#else
          float random_imag_a = rand() % (kRandomMax - kRandomMin) + kRandomMin;
          ac_complex<float> random_complex{random_real_a, random_imag_a};
          a_matrix[matrix_index * kMatrixSize + row * kColumns + col] =
              random_complex;
#endif

          float random_real_b = rand() % (kRandomMax - kRandomMin) + kRandomMin;
#if COMPLEX == 0
          b_matrix[matrix_index * kMatrixSize + row * kColumns + col] =
              random_real_b;
#else
          float random_imag_b = rand() % (kRandomMax - kRandomMin) + kRandomMin;
          ac_complex<float> random_complex{random_real_b, random_imag_b};
          b_matrix[matrix_index * kMatrixSize + row * kColumns + col] =
              random_complex;
#endif
        }  // end of col
      }    // end of row

#ifdef DEBUG
      std::cout << "A MATRIX " << matrix_index << std::endl;
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          std::cout << a_matrix[matrix_index * kMatrixSize + row * kColumns + col]
                    << " ";
        }  // end of col
        std::cout << std::endl;
      }  // end of row

      std::cout << "B MATRIX " << matrix_index << std::endl;
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          std::cout << b_matrix[matrix_index * kMatrixSize + row * kColumns + col]
                    << " ";
        }  // end of col
        std::cout << std::endl;
      }  // end of row

      std::cout << "B TRANSPOSED MATRIX " << matrix_index << std::endl;
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          std::cout << b_matrix[matrix_index * kMatrixSize + col * kColumns + row]
                    << " ";
        }  // end of col
        std::cout << std::endl;
      }  // end of row
#endif

    }  // end of matrix_index

    std::cout << "Running the matrix multiplication of " << kMatricesToMultiply
              << " matri" << (kMatricesToMultiply > 1 ? "ces " : "x ")
              << repetitions << " times" << std::endl;

    MatrixMultiplication(a_matrix, b_matrix, mm_matrix, q, kMatricesToMultiply,
                         repetitions);

    // For output post-processing (op)
    T mm_matrix_op[kRows][kColumns];

    // Floating-point error threshold value at which we decide that the design
    // computed an incorrect value
    constexpr float kErrorThreshold = 1e-4;

    // Check the MM matrices
    std::cout << "Verifying results on matrix ";
    for (int matrix_index = 0; matrix_index < kMatricesToMultiply;
         matrix_index++) {
      std::cout << matrix_index << std::endl;

      // keep track of MM element indexes
      size_t mm_idx = 0;

      // Read the MM matrix from the output vector to the mm_matrix_op matrix
      for (size_t i = 0; i < kRows; i++) {
        for (size_t j = 0; j < kColumns; j++) {
          mm_matrix_op[i][j] = mm_matrix[matrix_index * kMatrixSize + mm_idx];
          mm_idx++;
        }
      }

#ifdef DEBUG
      std::cout << "MM MATRIX" << std::endl;
      for (size_t i = 0; i < kRows; i++) {
        for (size_t j = 0; j < kColumns; j++) {
          std::cout << mm_matrix_op[i][j] << " ";
        }
        std::cout << std::endl;
      }
#endif

      // Count the number of errors found for this matrix
      size_t error_count = 0;
      bool error = false;

      for (size_t i = 0; i < kRows; i++) {
        for (size_t j = 0; j < kColumns; j++) {
          // Compute A * B at index i,j
          T axb_ij{0};
          for (size_t k = 0; k < kColumns; k++) {
            axb_ij += a_matrix[matrix_index * kMatrixSize + i * kColumns + k] *
                      b_matrix[matrix_index * kMatrixSize + j * kColumns + k];
          }

          // Verify that all the results are OK:
          // A * B = MM at index i,j
          bool axb_eq_mm;

#if COMPLEX == 0
          axb_eq_mm = abs(mm_matrix_op[i][j] - axb_ij) < kErrorThreshold;

#else
          axb_eq_mm =
              (abs(mm_matrix_op[i][j].r() - axb_ij.r()) < kErrorThreshold) &&
              (abs(mm_matrix_op[i][j].i() - axb_ij.i()) < kErrorThreshold);
#endif

          // If any of the checks failed
          if (!axb_eq_mm || !IsFinite(mm_matrix_op[i][j])) {
            // Increase the error count for this matrix
            error_count++;

            // Continue counting the errors even if we now we are going to
            // produce an error
            if (error) {
              continue;
            }

            if (!axb_eq_mm) {
              std::cout << "Error: (A*B)[" << i << "][" << j << "] = "
                        << axb_ij
                        << " but MM[" << i << "][" << j << "] = " 
                        << mm_matrix_op[i][j]
                        << std::endl;
            }
            if (!IsFinite(mm_matrix_op[i][j])) {
              std::cout << "MM[" << i << "][" << j << "] = " 
                        << mm_matrix_op[i][j]
                        << " is not finite" << std::endl;
            }
            error = true;
          }
        }  // end of j
      }    // end of i

      if (error_count > 0) {
        std::cout << std::endl << "FAILED" << std::endl;
        std::cout << std::endl
                  << "!!!!!!!!!!!!!! " << error_count << " errors" << std::endl;
        return 1;
      }
    }  // end of matrix_index

    std::cout << std::endl << "PASSED" << std::endl;
    return 0;

  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::cerr << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that is "
                 "set up correctly"
              << std::endl;
    std::cerr << "   If you are targeting the FPGA emulator, compile with "
                 "-DFPGA_EMULATOR"
              << std::endl;

    std::terminate();
  } catch (std::bad_alloc const &e) {
    std::cerr << "Caught a memory allocation exception on the host: "
              << e.what() << std::endl;
    std::cerr << "   You can reduce the memory requirement by reducing the "
                 "number of matrices generated. Specify a smaller number when "
                 "running the executable."
              << std::endl;
    std::cerr << "   In this run, more than "
              << (kMatrixSize * 3 * kMatricesToMultiply *
                  sizeof(float)) /
                     pow(2, 30)
              << " GBs of memory was requested for the decomposition of a "
              << "matrix of size " << kRows << " x " << kColumns << std::endl;
    std::terminate();
  }
}  // end of main
