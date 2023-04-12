#include <math.h>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <list>

#include "exception_handler.hpp"

#include "eigen.hpp"

#define DEBUG

#ifdef FPGA_SIMULATOR
#define SIZE_V 8
#else
#define SIZE_V SIZE
#endif

/*
  COLS_COMPONENT, ROWS_COMPONENT and FIXED_ITERATIONS are defined
  by the build system.

  Function arguments:
  - a_matrix:    The input matrix. Interpreted as a transposed matrix.
  - q_matrix:    The Q matrix. The function will overwrite this matrix.
  - eigen_values_matrix     The R matrix. The function will overwrite this matrix.
                 The vector will only contain the upper triangular elements
                 of the matrix, in a row by row fashion.
  - q:           The device queue.
  - matrix_count: Number of matrices to decompose.
  - repetitions: The number of repetitions of the computation to execute.
                 (for performance evaluation)
*/
// Real single precision floating-point QR Decomposition
void Eigen(std::vector<float> &a_matrix, std::vector<float> &q_matrix,
                     std::vector<float> &eigen_values_matrix, sycl::queue &q,
                     int matrix_count,
                     int repetitions) {
  EigenImpl<SIZE_V, FIXED_ITERATIONS, float>(a_matrix, q_matrix, eigen_values_matrix, q,
                                          matrix_count, repetitions);
}


/*
  returns if the given value is finite
*/
bool IsFinite(float val) { return std::isfinite(val); }

template<size_t size>
void QRD(std::vector<double> input, std::vector<double> &q, std::vector<double> &r){

  for (int i = 0; i<size; i++){
    // compute the norm of input_i
    double norm = 0;
    for (int k=0; k<size; k++){
      norm += input[k + i*size] * input[k + i*size];
    }
    double rii = std::sqrt(norm);
    r[i + i*size] = rii;

    for (int k=0; k<size; k++){
      q[k + i*size] = input[k + i*size] / rii;
    }  

    for (int j=i+1; j<size; j++){

      double rij = 0;
      for (int k=0; k<size; k++){
        rij += input[k + j*size] * q[k + i*size];
      }
      r[i + j*size] = rij;

      for (int k=0; k<size; k++){
        input[k + j*size] = input[k + j*size] - rij * q[k + i*size];
      }
    }     
  }

}

void Hessenberg(std::vector<std::vector<float>> &A) {
// [H,Q] = hessred(A)
// %
// % Compute the Hessenberg decomposition H = Q’*A*Q using
// % Householder transformations.
// %
// function [H,Q] = hessred(A)
// n = length(A);
// Q = eye(n); % Orthogonal transform so far
//  H = A; % Transformed matrix so far

//  for j = 1:n-2

//  % -- Find W = I-2vv’ to put zeros below H(j+1,j)
//  u = H(j+1:end,j);
//  u(1) = u(1) + sign(u(1))*norm(u);
//  v = u/norm(u);

//  % -- H := WHW’, Q := QW
//  H(j+1:end,:) = H(j+1:end,:)-2*v*(v’*H(j+1:end,:));
//  H(:,j+1:end) = H(:,j+1:end)-(H(:,j+1:end)*(2*v))*v’;
//  Q(:,j+1:end) = Q(:,j+1:end)-(Q(:,j+1:end)*(2*v))*v’;

//  end

// end
  // TODO
}


int main(int argc, char *argv[]) {
  constexpr size_t kRandomSeed = 1138;
  constexpr size_t kRandomMin = 1;
  constexpr size_t kRandomMax = 10;
  constexpr size_t kSize = SIZE_V;
  constexpr size_t kAMatrixSize = kSize * kSize;
  constexpr size_t kQMatrixSize = kSize * kSize;
  constexpr size_t kRMatrixSize = kSize;
  constexpr size_t kQRMatrixSize = kQMatrixSize + kRMatrixSize;

#if defined(FPGA_SIMULATOR)
  std::cout << "Using 32x32 matrices for simulation to reduce runtime" << std::endl;
#endif

  constexpr bool kShift = true;

  // Get the number of times we want to repeat the decomposition
  // from the command line.
// #if defined(FPGA_EMULATOR)
//   int repetitions = argc > 1 ? atoi(argv[1]) : 16;
// #elif defined(FPGA_SIMULATOR)
//   int repetitions = argc > 1 ? atoi(argv[1]) : 1;
// #else
//   int repetitions = argc > 1 ? atoi(argv[1]) : 819200;
// #endif

  int repetitions = 1;


  if (repetitions < 1) {
    std::cout << "Number of repetitions given is lower that 1." << std::endl;
    std::cout << "The decomposition must occur at least 1 time." << std::endl;
    std::cout << "Increase the number of repetitions (e.g. 16)." << std::endl;
    return 1;
  }

  constexpr size_t kMatricesToDecompose = 1;
// #if defined(FPGA_SIMULATOR)
//   constexpr size_t kMatricesToDecompose = 1;
// #else
//   constexpr size_t kMatricesToDecompose = 8;
// #endif

  try {

#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    // Enable the queue profiling to time the execution
    sycl::property_list
                    queue_properties{sycl::property::queue::enable_profiling()};
    sycl::queue q = sycl::queue(selector,
                                fpga_tools::exception_handler,
                                queue_properties);

    sycl::device device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // Create vectors to hold all the input and output matrices
    std::vector<float> a_matrix;
    std::vector<float> q_matrix;
    std::vector<float> eigen_values_matrix;

    a_matrix.resize(kAMatrixSize * kMatricesToDecompose);
    q_matrix.resize(kQMatrixSize * kMatricesToDecompose);
    eigen_values_matrix.resize(kRMatrixSize * kMatricesToDecompose);

    std::cout << "Generating " << kMatricesToDecompose << " random real ";
    std::cout << "matri" << (kMatricesToDecompose > 1 ? "ces" : "x")
              << " of size "
              << kSize << "x" << kSize << " " << std::endl;

    // Generate the random symetric matrix input matrices
    srand(kRandomSeed);
    float expected_eigen_values[kSize*kMatricesToDecompose];

    int total_iterations = 0;
    constexpr double kThresholdMatrixGeneration = 1e-4;

    for(int matrix_index = 0; matrix_index < kMatricesToDecompose;
                                                                matrix_index++){

      std::cout << "Generating matrix " << matrix_index << std::endl;
      for (size_t row = 0; row < kSize; row++) {
        for (size_t col = row; col < kSize; col++) {
          float random_real;
          if (col > (row+1)) {
            random_real = 0;
          }
          else{
            random_real = rand() % (kRandomMax - kRandomMin) + kRandomMin;
          }
          a_matrix[matrix_index * kAMatrixSize
                 + col * kSize + row] = random_real;
          a_matrix[matrix_index * kAMatrixSize
                 + row * kSize + col] = random_real;
        }  // end of col
      }    // end of row


  // #ifdef DEBUG
      std::cout << "A MATRIX " << matrix_index << std::endl;
      for (size_t row = 0; row < kSize; row++) {
        for (size_t col = 0; col < kSize; col++) {
          std::cout << a_matrix[matrix_index * kAMatrixSize
                              + col * kSize + row] << " ";
        }  // end of col
        std::cout << std::endl;
      }  // end of row
  // #endif

      int iterations = 0;
      bool cond = false;
      std::vector<double> q, r, rq;
      q.resize(kSize*kSize);
      r.resize(kSize*kSize);
      rq.resize(kSize*kSize);

      for(int k = 0; k< kSize*kSize; k++){
        rq[k] = a_matrix[matrix_index * kAMatrixSize + k];
      }

      while (!cond){
        for (int k = 0; k<kSize*kSize; k++){
          r[k] = 0;
        }
        
        double shift_value = 0;

        if constexpr (kShift){
          // First find where the shift should be applied
          // Start from the last submatrix
          int shift_row = kSize-2;
          for(int row=kSize-1; row>=1; row--){
            bool row_is_zero = true;
            for (int col=0; col<row; col++){
              row_is_zero &= (fabs(rq[row + kSize*col]) < kThresholdMatrixGeneration);
            }
            if (!row_is_zero){
              break;
            }
            shift_row--;
          }  

          if(shift_row>=0){
            // Compute the shift value
            // Take the submatrix:
            // [a b] 
            // [b c]
            // and compute the shift such as
            // mu = c - (sign(d)* b*b)/(abs(d) + sqrt(d*d + b*b))
            // where d = (a - c)/2

            double a = rq[shift_row + kSize*shift_row];
            double b = rq[shift_row + kSize*(shift_row+1)];
            double c = rq[(shift_row+1) + kSize*(shift_row+1)];

            double d = (a - c) / 2;
            double b_squared = b*b;
            double d_squared = d*d;
            double b_squared_signed = d<0 ? -b_squared : b_squared;
            shift_value = c - b_squared_signed / (abs(d) + sqrt(d_squared + b_squared));
          }

          // Subtract the shift value from the diagonal of RQ
          for(int row=0; row<kSize; row++){
            rq[row + kSize*row] -= shift_value; 
          }          

        }

        QRD<kSize>(rq, q, r);

        // Compute RQ
        for(int row=0; row<kSize; row++){
          for (int col=0; col<kSize; col++){
            double prod = 0;
            for (int k=0; k<kSize; k++){
              prod += r[row + kSize*k] * q[k + kSize*col]; 
            }
            rq[row + kSize*col] = prod; 
          }
        }

        if constexpr (kShift){
          // Add the shift value back to the diagonal of RQ
          for(int row=0; row<kSize; row++){
            rq[row + kSize*row] += shift_value; 
          }     
        }


        bool all_below_threshold = true;
        for(int row=1; row<kSize; row++){
          for(int col=0; col<(row); col++){
            all_below_threshold &= (rq[row + kSize*col] < kThresholdMatrixGeneration);
          }
        }

        cond = all_below_threshold;

      // #ifdef DEBUG
          // std::cout << "Q MATRIX " << matrix_index << std::endl;
          // for (size_t row = 0; row < kSize; row++) {
          //   for (size_t col = 0; col < kSize; col++) {
          //     std::cout << q[col * kSize + row] << " ";
          //   }  // end of col
          //   std::cout << std::endl;
          // }  // end of row

          // std::cout << "R MATRIX " << matrix_index << std::endl;
          // for (size_t row = 0; row < kSize; row++) {
          //   for (size_t col = 0; col < kSize; col++) {
          //     std::cout << r[col * kSize + row] << " ";
          //   }  // end of col
          //   std::cout << std::endl;
          // }  // end of row

          // std::cout << "RQ MATRIX " << matrix_index << std::endl;
          // for (size_t row = 0; row < kSize; row++) {
          //   for (size_t col = 0; col < kSize; col++) {
          //     std::cout << rq[col * kSize + row] << " ";
          //   }  // end of col
          //   std::cout << std::endl;
          // }  // end of row
      // #endif

          iterations++;
          if(iterations>(kSize*kSize*kSize*16)){
            std::cout << "Number of iterations too high" << std::endl; 
            break;
          }
      }

      if(iterations>kSize*kSize*kSize*16){
        matrix_index--;
      }
      else{
        total_iterations += iterations;
        // std::cout << "expected eigen values after " << iterations << " iterations:" << std::endl;
        for(int k=0; k<kSize; k++){
          // std::cout << rq[k + kSize*k] << " ";
          expected_eigen_values[k + matrix_index*kSize] = rq[k + kSize*k]; 
        }
        // std::cout << std::endl;
      }

    } // end of matrix_index

    std::cout << "Average number of iterations using the regular QR iterations: " << total_iterations/kMatricesToDecompose << std::endl;

    std::cout << "Running QR decomposition of " << kMatricesToDecompose
              << " matri" << (kMatricesToDecompose > 1 ? "ces " : "x ")
              << repetitions << " times" << std::endl;

    Eigen(a_matrix, q_matrix, eigen_values_matrix, q, kMatricesToDecompose, repetitions);

    // For output post-processing (op)
    float q_matrix_op[kSize][kSize];
    float eigen_values_matrix_op[kSize];

    // Floating-point error threshold value at which we decide that the design
    // computed an incorrect value
    constexpr float kErrorThreshold = 1e-3;

    // Count the number of errors found for this matrix
    size_t error_count = 0;

    // Check Q and R matrices
    std::cout << "Verifying results..." << std::endl;
    for(int matrix_index = 0; matrix_index < kMatricesToDecompose;
                                                                matrix_index++){

      // keep track of Q element index
      size_t q_idx = 0;

      // Read the R matrix from the output vector to the RMatrixOP matrix
      for (size_t i = 0; i < kSize; i++) {
        eigen_values_matrix_op[i] =
            eigen_values_matrix[matrix_index * kRMatrixSize + i];
      }

      // Read the Q matrix from the output vector to the QMatrixOP matrix
      for (size_t j = 0; j < kSize; j++) {
        for (size_t i = 0; i < kSize; i++) {
          q_matrix_op[i][j] = q_matrix[matrix_index*kQMatrixSize
                                     + q_idx];
          q_idx++;
        }
      }

  // #ifdef DEBUG
  //     std::cout << "Eigen values MATRIX" << std::endl;
  //     for (size_t i = 0; i < kSize; i++) {
  //       std::cout << eigen_values_matrix_op[i] << " ";
  //     }
  //     std::cout << std::endl;

  //     // std::cout << "Q MATRIX" << std::endl;
  //     // for (size_t i = 0; i < kSize; i++) {
  //     //   for (size_t j = 0; j < kSize; j++) {
  //     //     std::cout << q_matrix_op[i][j] << " ";
  //     //   }
  //     //   std::cout << std::endl;
  //     // }
  // #endif

      for (size_t i = 0; i < kSize; i++) {

        bool found_a_matching_eigen_value = fabs(fabs(eigen_values_matrix_op[i]) -fabs(expected_eigen_values[matrix_index * kSize + i])) <= kErrorThreshold;

        // It may be that the eigen values are not in the same order.
        if(!found_a_matching_eigen_value) {

          for (size_t j = 0; j < kSize; j++) {
            if(fabs(fabs(eigen_values_matrix_op[i]) - fabs(expected_eigen_values[matrix_index * kSize + j])) <= kErrorThreshold){
              found_a_matching_eigen_value = true;
              break;
            }
          }
        }

        if(!found_a_matching_eigen_value){
          std::cout << "Eigen value computed " << eigen_values_matrix_op[i] 
                    << " does not match any of the precomputed Eigen values ";
          for (int i=0; i<kSize; i++) {
            std::cout << expected_eigen_values[matrix_index * kSize + i] << " ";
          }

          std::cout << std::endl;
          std::cout << "Input matrix " << matrix_index << std::endl;
          for (size_t row = 0; row < kSize; row++) {
            for (size_t col = 0; col < kSize; col++) {
              std::cout << a_matrix[matrix_index * kAMatrixSize
                                  + col * kSize + row] << " ";
            }  // end of col
            std::cout << std::endl;
          }  // end of row
          std::cout << "Precomputed Eigen values " << std::endl;
          for (size_t i = 0; i < kSize; i++) {
            std::cout << expected_eigen_values[matrix_index * kSize + i] << " ";
          }
          std::cout << std::endl;

          std::cout << "Returned Eigen values " << std::endl;
          for (size_t i = 0; i < kSize; i++) {
            std::cout << eigen_values_matrix_op[i] << " ";
          }
          std::cout << std::endl;

          error_count++;
        }
        if(!IsFinite(eigen_values_matrix_op[i])){
          error_count++;          
        }
      }

      if (error_count > 0) {
        std::cout << std::endl << "FAILED" << std::endl;
        std::cout << std::endl
                  << "!!!!!!!!!!!!!! " << error_count << " errors" << std::endl;
        return 1;
      }

    } // end of matrix_index


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
              << ((kAMatrixSize + kQRMatrixSize) * 2 * kMatricesToDecompose
                 * sizeof(float)) / pow(2, 30)
              << " GBs of memory was requested for the decomposition of a "
              << "matrix of size " << kSize << " x " << kSize
              << std::endl;
    std::terminate();
  }
}  // end of main
