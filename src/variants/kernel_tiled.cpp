#include "kernel_tiled.hpp"

#include <chrono>

#include <Vc/Vc>
#include <boost/align/aligned_allocator.hpp>

// best parameters
// #define L3_X 420 // max 2 L3 par set to 1024 (rest 512)
// #define L3_Y 256
// #define L3_K_STEP 512
#define L3_X 420 // max 2 L3 par set to 1024 (rest 512)
#define L3_Y 256
#define L3_K_STEP 256

// parameters for kernel tuning
// #define L3_X 70 // max 2 L3 par set to 1024 (rest 512)
// #define L3_Y 64
// #define L3_K_STEP 64
#define L2_X 70 // max 2 L2 par set to 128 (rest 64)
#define L2_Y 64
#define L2_K_STEP 128
#define L1_X 35 // max all L1 par set to 32
#define L1_Y 16
#define L1_K_STEP 64
#define X_REG 5 // cannot be changed!
#define Y_REG 8 // cannot be changed!
#define K_REG 1 // cannot be changed!

// 4x12 approach kernel tuning
// #define L3_X 32 // max 2 L3 par set to 1024 (rest 512)
// #define L3_Y 24
// #define L3_K_STEP 32
// #define L2_X 32 // max 2 L2 par set to 128 (rest 64)
// #define L2_Y 24
// #define L2_K_STEP 32
// #define L1_X 32 // max all L1 par set to 32
// #define L1_Y 24
// #define L1_K_STEP 32
// #define X_REG 4 // cannot be changed!
// #define Y_REG 12 // cannot be changed!
// #define K_REG 1 // cannot be changed!

namespace kernel_tiled {

void kernel_tiled::verify_blocking_setup() {
  if (!((L2_X % L1_X == 0) && (L3_X % L2_X == 0))) {
    std::cout << "error: x direction blocking not set up correctly"
              << std::endl;
    throw;
  }
  if (!((L2_Y % L1_Y == 0) && (L3_Y % L2_Y == 0))) {
    std::cout << "error: y direction blocking not set up correctly"
              << std::endl;
    throw;
  }
  if (!((L2_K_STEP % L1_K_STEP == 0) && (L3_K_STEP % L2_K_STEP == 0))) {
    std::cout << "error: k direction blocking not set up correctly"
              << std::endl;
    throw;
  }
}

kernel_tiled::kernel_tiled(size_t N, std::vector<double> &A_org,
                           std::vector<double> &B_org, bool transposed,
                           uint64_t repetitions, uint64_t verbose)
    : N_org(N), repetitions(repetitions), verbose(verbose) {
  verify_blocking_setup();

  // k direction padding
  size_t k_pad = L3_K_STEP - (N % L3_K_STEP);
  if (k_pad == L3_K_STEP) {
    k_pad = 0; // nothing to pad
  }
  size_t x_pad = L3_X - (N % L3_X);
  if (x_pad == L3_X) {
    x_pad = 0; // nothing to pad
  }
  size_t y_pad = L3_Y - (N % L3_Y);
  if (y_pad == L3_Y) {
    y_pad = 0; // nothing to pad
  }

  if (verbose >= 1) {
    std::cout << "matrix padding: x_pad = " << x_pad << ", y_pad = " << y_pad
              << ", k_pad = " << k_pad << std::endl;
  }

  X_size = N + x_pad;
  Y_size = N + y_pad;
  K_size = N + k_pad;

  if (verbose >= 1) {
    std::cout << "matrix dimensions for calculation: X = " << X_size
              << ", Y = " << Y_size << ", K = " << K_size << std::endl;
  }

  A = std::vector<double>(X_size * K_size);
  std::fill(A.begin(), A.end(), 0.0);
  for (size_t x = 0; x < N_org; x++) {
    for (size_t k = 0; k < N_org; k++) {
      A.at(x * K_size + k) = A_org.at(x * N + k);
    }
  }
  B = std::vector<double>(K_size * Y_size);
  std::fill(B.begin(), B.end(), 0.0);
  for (size_t y = 0; y < N_org; y++) {
    for (size_t k = 0; k < N_org; k++) {
      B.at(k * Y_size + y) = B_org.at(k * N + y);
    }
  }
}

std::vector<double> kernel_tiled::matrix_multiply(double &duration) {

  // create a matrix of l1 cachable submatrices, caching by tiling, no large
  // strides even without padding
  std::vector<double, boost::alignment::aligned_allocator<double, 32>> C_padded(
      X_size * Y_size);
  std::fill(C_padded.begin(), C_padded.end(), 0.0);

  // create a matrix of l1 cachable submatrices, caching by tiling, no large
  // strides even without padding
  // is also padded if padding is enabled (row padded only)
  std::vector<double, boost::alignment::aligned_allocator<double, 32>> A_trans(
      K_size * X_size);
  for (size_t l1_x = 0; l1_x < X_size / L1_X; l1_x += 1) {
    for (size_t l1_k = 0; l1_k < K_size / L1_K_STEP; l1_k += 1) {
      size_t base_index = (L1_X * L1_K_STEP) *
                          (l1_k * (X_size / L1_X) + l1_x); // look up submatrix
      for (size_t x = 0; x < L1_X; x++) {
        for (size_t k = 0; k < L1_K_STEP; k++) {
          A_trans[base_index + k * L1_X + x] =
              A[(l1_x * L1_X + x) * K_size + (l1_k * L1_K_STEP + k)];
        }
      }
    }
  }

  // don't need padding for B, no dependency to row count
  std::vector<double, boost::alignment::aligned_allocator<double, 32>> B_padded(
      K_size * Y_size);
  for (size_t l1_y = 0; l1_y < (Y_size / L1_Y); l1_y += 1) {
    for (size_t l1_k = 0; l1_k < (K_size / L1_K_STEP); l1_k += 1) {
      size_t base_index = (L1_Y * L1_K_STEP) *
                          (l1_k * (Y_size / L1_Y) + l1_y); // look up submatrix
      for (size_t y = 0; y < L1_Y; y++) {
        for (size_t k = 0; k < L1_K_STEP; k++) {
          B_padded[base_index + k * L1_Y + y] =
              B[(l1_k * L1_K_STEP + k) * Y_size + (l1_y * L1_Y + y)];
        }
      }
    }
  }

  // std::cout << "A:" << std::endl;
  // for (uint64_t x = 0; x < N; x++) {
  //   for (uint64_t y = 0; y < N; y++) {
  // 	if (y > 0) {
  // 	  std::cout << ", ";
  // 	}
  // 	std::cout << A[x * N + y];
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << "A trans:" << std::endl;
  // for (uint64_t x = 0; x < N; x++) {
  //   for (uint64_t y = 0; y < N; y++) {
  // 	if (y > 0) {
  // 	  std::cout << ", ";
  // 	}
  // 	std::cout << A[y * N + x];
  //   }
  //   std::cout << std::endl;
  // }

  // std::cout << "A_trans (sub-matrices, DOESNT CONSIDER PADDING):" <<
  // std::endl;
  // size_t submatrix_index = 0;
  // for (uint64_t k = 0; k < N * N; k++) {
  //   if (submatrix_index < L1_X * L1_K_STEP) {
  // 	if (submatrix_index < L1_X * L1_K_STEP) {
  // 	  if (submatrix_index > 0) {
  // 	    std::cout << ", ";
  // 	  }
  // 	  std::cout << A_trans[k];
  // 	}
  // 	submatrix_index++;
  //   } else {
  // 	submatrix_index = 1;
  // 	std::cout << std::endl;
  // 	std::cout << A_trans[k];
  //   }
  // }
  // std::cout << std::endl;

  // std::cout << "B:" << std::endl;
  // for (uint64_t x = 0; x < N; x++) {
  //   for (uint64_t y = 0; y < N; y++) {
  // 	if (y > 0) {
  // 	  std::cout << ", ";
  // 	}
  // 	std::cout << B[x * N + y];
  //   }
  //   std::cout << std::endl;
  // }

  // std::cout << "B_padded (sub-matrices, DOESNT CONSIDER PADDING):" <<
  // std::endl;
  // submatrix_index = 0;
  // for (uint64_t k = 0; k < N * N; k++) {
  //   if (submatrix_index < L1_Y * L1_K_STEP) {
  // 	if (submatrix_index < L1_Y * L1_K_STEP) {
  // 	  if (submatrix_index > 0) {
  // 	    std::cout << ", ";
  // 	  }
  // 	  std::cout << B_padded[k];
  // 	}
  // 	submatrix_index++;
  //   } else {
  // 	submatrix_index = 1;
  // 	std::cout << std::endl;
  // 	std::cout << B_padded[k];
  //   }
  // }
  // std::cout << std::endl;

  for (size_t rep = 0; rep < repetitions; rep++) {

    std::fill(C_padded.begin(), C_padded.end(), 0.0);

    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();

    using Vc::double_v;
// L3 blocking and parallelization
#pragma omp parallel for collapse(2)
    for (size_t l3_x = 0; l3_x < X_size; l3_x += L3_X) {
      for (size_t l3_y = 0; l3_y < Y_size; l3_y += L3_Y) {
        for (size_t l3_k = 0; l3_k < K_size; l3_k += L3_K_STEP) {
          // L2 blocking
          for (size_t l2_x = l3_x; l2_x < l3_x + L3_X; l2_x += L2_X) {
            for (size_t l2_y = l3_y; l2_y < l3_y + L3_Y; l2_y += L2_Y) {
              for (size_t l2_k = l3_k; l2_k < l3_k + L3_K_STEP;
                   l2_k += L2_K_STEP) {
                // L1 blocking
                for (size_t l1_x = l2_x; l1_x < l2_x + L2_X; l1_x += L1_X) {
                  size_t l1_block_x = l1_x / L1_X;
                  for (size_t l1_y = l2_y; l1_y < l2_y + L2_Y; l1_y += L1_Y) {
                    size_t l1_block_y = l1_y / L1_Y;
                    size_t C_base_index =
                        (L1_X * L1_Y) *
                        (l1_block_x * (Y_size / L1_Y) + l1_block_y);
                    for (size_t l1_k = l2_k; l1_k < l2_k + L2_K_STEP;
                         l1_k += L1_K_STEP) {
                      size_t l1_block_k = l1_k / L1_K_STEP;
                      size_t A_base_index =
                          (L1_X * L1_K_STEP) *
                          (l1_block_k * (X_size / L1_X) + l1_block_x);
                      size_t B_base_index =
                          (L1_Y * L1_K_STEP) *
                          (l1_block_k * (Y_size / L1_Y) + l1_block_y);
                      // Register blocking
                      for (size_t x = 0; x < L1_X; x += X_REG) {
                        for (size_t y = 0; y < L1_Y; y += Y_REG) {

                          double_v acc_11 = 0.0;
                          double_v acc_21 = 0.0;
                          double_v acc_31 = 0.0;
                          double_v acc_41 = 0.0;

                          double_v acc_51 = 0.0;

                          double_v acc_12 = 0.0;
                          double_v acc_22 = 0.0;
                          double_v acc_32 = 0.0;
                          double_v acc_42 = 0.0;

                          double_v acc_52 = 0.0;

                          // comment in for 4x12 approach
                          // double_v acc_11 = 0.0;
                          // double_v acc_21 = 0.0;
                          // double_v acc_31 = 0.0;
                          // double_v acc_41 = 0.0;

                          // double_v acc_12 = 0.0;
                          // double_v acc_22 = 0.0;
                          // double_v acc_32 = 0.0;
                          // double_v acc_42 = 0.0;

                          // double_v acc_13 = 0.0;
                          // double_v acc_23 = 0.0;
                          // double_v acc_33 = 0.0;
                          // double_v acc_43 = 0.0;

                          for (size_t k_inner = 0; k_inner < L1_K_STEP;
                               k_inner += 1) {

                            double_v b_temp_1 = double_v(
                                &B_padded[B_base_index + k_inner * L1_Y + y],
                                Vc::flags::vector_aligned);
                            double_v b_temp_2 =
                                double_v(&B_padded[B_base_index +
                                                   k_inner * L1_Y + (y + 4)],
                                         Vc::flags::vector_aligned);

                            double_v a_temp_1 =
                                A_trans[A_base_index + k_inner * L1_X +
                                        (x + 0)];
                            double_v a_temp_2 =
                                A_trans[A_base_index + k_inner * L1_X +
                                        (x + 1)];
                            double_v a_temp_3 =
                                A_trans[A_base_index + k_inner * L1_X +
                                        (x + 2)];

                            acc_11 += a_temp_1 * b_temp_1;
                            acc_21 += a_temp_2 * b_temp_1;

                            acc_12 += a_temp_1 * b_temp_2;
                            acc_22 += a_temp_2 * b_temp_2;

                            double_v a_temp_4 =
                                A_trans[A_base_index + k_inner * L1_X +
                                        (x + 3)];
                            double_v a_temp_5 =
                                A_trans[A_base_index + k_inner * L1_X +
                                        (x + 4)];

                            acc_31 += a_temp_3 * b_temp_1;
                            acc_32 += a_temp_3 * b_temp_2;

                            acc_41 += a_temp_4 * b_temp_1;
                            acc_51 += a_temp_5 * b_temp_1;

                            acc_42 += a_temp_4 * b_temp_2;
                            acc_52 += a_temp_5 * b_temp_2;

                            // comment in for 4x12 approach
                            // double_v b_temp_1 =
                            //   double_v(&B_padded[B_base_index + k_inner *
                            //   L1_Y + y], Vc::flags::vector_aligned);
                            // double_v b_temp_2 =
                            //   double_v(&B_padded[B_base_index + k_inner *
                            //   L1_Y + (y + 4)], Vc::flags::vector_aligned);
                            // double_v b_temp_3 =
                            //   double_v(&B_padded[B_base_index + k_inner *
                            //   L1_Y + (y + 8)], Vc::flags::vector_aligned);

                            // double_v a_temp_1 = A_trans[A_base_index +
                            // k_inner * L1_X + (x + 0)];
                            // double_v a_temp_2 = A_trans[A_base_index +
                            // k_inner * L1_X + (x + 1)];
                            // double_v a_temp_3 = A_trans[A_base_index +
                            // k_inner * L1_X + (x + 2)];
                            // double_v a_temp_4 = A_trans[A_base_index +
                            // k_inner * L1_X + (x + 3)];

                            // acc_11 += a_temp_1 * b_temp_1;
                            // acc_12 += a_temp_1 * b_temp_2;
                            // acc_13 += a_temp_1 * b_temp_3;

                            // acc_21 += a_temp_2 * b_temp_1;
                            // acc_22 += a_temp_2 * b_temp_2;
                            // acc_23 += a_temp_2 * b_temp_3;

                            // acc_31 += a_temp_3 * b_temp_1;
                            // acc_32 += a_temp_3 * b_temp_2;
                            // acc_33 += a_temp_3 * b_temp_3;

                            // acc_41 += a_temp_4 * b_temp_1;
                            // acc_42 += a_temp_4 * b_temp_2;
                            // acc_43 += a_temp_4 * b_temp_3;
                          }

                          double_v res_11 = double_v(
                              &C_padded[C_base_index + (x + 0) * L1_Y + y],
                              Vc::flags::element_aligned);
                          res_11 += acc_11;
                          res_11.memstore(
                              &C_padded[C_base_index + (x + 0) * L1_Y + y],
                              Vc::flags::element_aligned);
                          double_v res_21 = double_v(
                              &C_padded[C_base_index + (x + 1) * L1_Y + y],
                              Vc::flags::element_aligned);
                          res_21 += acc_21;
                          res_21.memstore(
                              &C_padded[C_base_index + (x + 1) * L1_Y + y],
                              Vc::flags::element_aligned);
                          double_v res_31 = double_v(
                              &C_padded[C_base_index + (x + 2) * L1_Y + y],
                              Vc::flags::element_aligned);
                          res_31 += acc_31;
                          res_31.memstore(
                              &C_padded[C_base_index + (x + 2) * L1_Y + y],
                              Vc::flags::element_aligned);
                          double_v res_41 = double_v(
                              &C_padded[C_base_index + (x + 3) * L1_Y + y],
                              Vc::flags::element_aligned);
                          res_41 += acc_41;
                          res_41.memstore(
                              &C_padded[C_base_index + (x + 3) * L1_Y + y],
                              Vc::flags::element_aligned);

                          // has to be commented out for 4x12 approach
                          double_v res_51 = double_v(
                              &C_padded[C_base_index + (x + 4) * L1_Y + y],
                              Vc::flags::element_aligned);
                          res_51 += acc_51;
                          res_51.memstore(
                              &C_padded[C_base_index + (x + 4) * L1_Y + y],
                              Vc::flags::element_aligned);

                          double_v res_12 =
                              double_v(&C_padded[C_base_index + (x + 0) * L1_Y +
                                                 (y + 4)],
                                       Vc::flags::element_aligned);
                          res_12 += acc_12;
                          res_12.memstore(&C_padded[C_base_index +
                                                    (x + 0) * L1_Y + (y + 4)],
                                          Vc::flags::element_aligned);
                          double_v res_22 =
                              double_v(&C_padded[C_base_index + (x + 1) * L1_Y +
                                                 (y + 4)],
                                       Vc::flags::element_aligned);
                          res_22 += acc_22;
                          res_22.memstore(&C_padded[C_base_index +
                                                    (x + 1) * L1_Y + (y + 4)],
                                          Vc::flags::element_aligned);
                          double_v res_32 =
                              double_v(&C_padded[C_base_index + (x + 2) * L1_Y +
                                                 (y + 4)],
                                       Vc::flags::element_aligned);
                          res_32 += acc_32;
                          res_32.memstore(&C_padded[C_base_index +
                                                    (x + 2) * L1_Y + (y + 4)],
                                          Vc::flags::element_aligned);
                          double_v res_42 =
                              double_v(&C_padded[C_base_index + (x + 3) * L1_Y +
                                                 (y + 4)],
                                       Vc::flags::element_aligned);
                          res_42 += acc_42;
                          res_42.memstore(&C_padded[C_base_index +
                                                    (x + 3) * L1_Y + (y + 4)],
                                          Vc::flags::element_aligned);

                          // has to be commented out for 4x12 approach
                          double_v res_52 =
                              double_v(&C_padded[C_base_index + (x + 4) * L1_Y +
                                                 (y + 4)],
                                       Vc::flags::element_aligned);
                          res_52 += acc_52;
                          res_52.memstore(&C_padded[C_base_index +
                                                    (x + 4) * L1_Y + (y + 4)],
                                          Vc::flags::element_aligned);

                          // has to be commented in for 4x12 approach
                          // double_v res_13 = double_v(&C_padded[C_base_index +
                          // (x + 0) * L1_Y + (y + 8)]);
                          // res_13 += acc_13;
                          // res_13.memstore(&C_padded[C_base_index + (x + 0) *
                          // L1_Y + (y + 8)]);
                          // double_v res_23 = double_v(&C_padded[C_base_index +
                          // (x + 1) * L1_Y + (y + 8)]);
                          // res_23 += acc_23;
                          // res_23.memstore(&C_padded[C_base_index + (x + 1) *
                          // L1_Y + (y + 8)]);
                          // double_v res_33 = double_v(&C_padded[C_base_index +
                          // (x + 2) * L1_Y + (y + 8)]);
                          // res_33 += acc_33;
                          // res_33.memstore(&C_padded[C_base_index + (x + 2) *
                          // L1_Y + (y + 8)]);
                          // double_v res_43 = double_v(&C_padded[C_base_index +
                          // (x + 3) * L1_Y + (y + 8)]);
                          // res_43 += acc_43;
                          // res_43.memstore(&C_padded[C_base_index + (x + 3) *
                          // L1_Y + (y + 8)]);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double>(end - start).count();
  }

  std::cout << "duration inner: " << duration << "s" << std::endl;

  // std::cout << "C_padded result (sub-matrices, DOESNT CONSIDER PADDING):" <<
  // std::endl;
  // submatrix_index = 0;
  // for (uint64_t k = 0; k < N * N; k++) {
  //   if (submatrix_index < L1_Y * L1_X) {
  // 	if (submatrix_index < L1_Y * L1_X) {
  // 	  if (submatrix_index > 0) {
  // 	    std::cout << ", ";
  // 	  }
  // 	  std::cout << C_padded[k];
  // 	}
  // 	submatrix_index++;
  //   } else {
  // 	submatrix_index = 1;
  // 	std::cout << std::endl;
  // 	std::cout << C_padded[k];
  //   }
  // }
  // std::cout << std::endl;

  std::vector<double> C_return(N_org * N_org);
  std::fill(C_return.begin(), C_return.end(), 0.0);

  for (size_t l1_x = 0; l1_x < X_size / L1_X; l1_x += 1) {
    for (size_t l1_y = 0; l1_y < Y_size / L1_Y; l1_y += 1) {
      size_t base_index =
          (L1_X * L1_Y) * (l1_x * (Y_size / L1_Y) + l1_y); // look up submatrix
      for (size_t x = 0; x < L1_X; x++) {
        for (size_t y = 0; y < L1_Y; y++) {
          // skip padding
          if (l1_x * L1_X + x < N_org && l1_y * L1_Y + y < N_org) {
            C_return.at((l1_x * L1_X + x) * N_org + (l1_y * L1_Y + y)) =
                C_padded.at(base_index + x * L1_Y + y);
          }
        }
      }
    }
  }

  double flops = 2 * static_cast<double>(X_size) * static_cast<double>(Y_size) *
                 static_cast<double>(K_size);
  double gflop = flops / 1E9;
  std::cout << "[X_size = " << X_size << ", Y_size = " << Y_size
            << ", K_size = " << K_size
            << "] inner performance: " << (repetitions * gflop / duration)
            << "Gflops (average across repetitions)" << std::endl;

  return C_return;
}
}

#undef L3_X
#undef L3_Y
#undef L3_K_STEP
#undef L2_X
#undef L2_Y
#undef L2_K_STEP
#undef L1_X
#undef L1_Y
#undef L1_K_STEP
#undef X_REG
#undef Y_REG
