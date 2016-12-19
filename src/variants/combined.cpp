#include "combined.hpp"

#include <chrono>

#include "index_iterator.hpp"

#include <hpx/include/iostreams.hpp>

#include <Vc/Vc>
using Vc::double_v;
#include <boost/align/aligned_allocator.hpp>

// best parameters
// #define L3_X 420 // max 2 L3 par set to 1024 (rest 512)
// #define L3_Y 256
// #define L3_K_STEP 512
#define L3_X 420 // max 2 L3 par set to 1024 (rest 512)
#define L3_Y 256
#define L3_K_STEP 256
// #define L3_X 840 // max 2 L3 par set to 1024 (rest 512)
// #define L3_Y 512
// #define L3_K_STEP 1024
// #define L3_X 70 // max 2 L3 par set to 1024 (rest 512)
// #define L3_Y 64
// #define L3_K_STEP 128

// parameters for kernel tuning
// #define L3_X 70 // max 2 L3 par set to 1024 (rest 512)
// #define L3_Y 64
// #define L3_K_STEP 64
#define L2_X 70 // max 2 L2 par set to 128 (rest 64)
#define L2_Y 64
#define L2_K_STEP 128
// #define L2_X 35 // max 2 L2 par set to 128 (rest 64)
// #define L2_Y 16
// #define L2_K_STEP 64
#define L1_X 35 // max all L1 par set to 32
#define L1_Y 16
#define L1_K_STEP 64
#define X_REG 5 // cannot be changed!
#define Y_REG 8 // cannot be changed!
#define K_REG 1 // cannot be changed!

using namespace index_iterator;

namespace combined {

void combined::verify_blocking_setup() {
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

combined::combined(size_t N, std::vector<double> &A_org,
                   std::vector<double> &B_org, bool transposed,
                   uint64_t block_result, uint64_t block_input,
                   uint64_t repetitions, uint64_t verbose)
    : N_org(N), A(A_org), B(B_org), transposed(transposed),
      block_result(block_result), block_input(block_input),
      repetitions(repetitions), verbose(verbose) {
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

std::vector<double> combined::matrix_multiply(double &duration) {

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

  std::vector<size_t> min = {0, 0, 0};
  std::vector<size_t> max = {X_size, Y_size, K_size};

  for (size_t rep = 0; rep < repetitions; rep++) {

    blocking_pseudo_execution_policy<size_t> policy(3);
    // specify with ascending cache level
    policy.set_final_steps({L1_X, L1_Y, L1_K_STEP});
    policy.add_blocking({L2_X, L2_Y, L2_K_STEP}, {false, false, false});
    policy.add_blocking({L3_X, L3_Y, L3_K_STEP},
                        {true, true, false}); // LLC blocking

    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();

    iterate_indices<3>(
        policy, min, max, [&C_padded, &A_trans, &B_padded,
                           this](size_t l1_x, size_t l1_y, size_t l1_k) {
          size_t l1_block_x = l1_x / L1_X;
          size_t l1_block_y = l1_y / L1_Y;
          size_t C_base_index =
              (L1_X * L1_Y) * (l1_block_x * (Y_size / L1_Y) + l1_block_y);
          size_t l1_block_k = l1_k / L1_K_STEP;
          size_t A_base_index =
              (L1_X * L1_K_STEP) * (l1_block_k * (X_size / L1_X) + l1_block_x);
          size_t B_base_index =
              (L1_Y * L1_K_STEP) * (l1_block_k * (Y_size / L1_Y) + l1_block_y);
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

              for (size_t k_inner = 0; k_inner < L1_K_STEP; k_inner += 1) {

                double_v b_temp_1 = double_v(
                    &B_padded[B_base_index + k_inner * L1_Y + y], Vc::Aligned);
                double_v b_temp_2 =
                    double_v(&B_padded[B_base_index + k_inner * L1_Y + (y + 4)],
                             Vc::Aligned);

                double_v a_temp_1 =
                    A_trans[A_base_index + k_inner * L1_X + (x + 0)];
                double_v a_temp_2 =
                    A_trans[A_base_index + k_inner * L1_X + (x + 1)];
                double_v a_temp_3 =
                    A_trans[A_base_index + k_inner * L1_X + (x + 2)];

                acc_11 += a_temp_1 * b_temp_1;
                acc_21 += a_temp_2 * b_temp_1;

                acc_12 += a_temp_1 * b_temp_2;
                acc_22 += a_temp_2 * b_temp_2;

                double_v a_temp_4 =
                    A_trans[A_base_index + k_inner * L1_X + (x + 3)];
                double_v a_temp_5 =
                    A_trans[A_base_index + k_inner * L1_X + (x + 4)];

                acc_31 += a_temp_3 * b_temp_1;
                acc_32 += a_temp_3 * b_temp_2;

                acc_41 += a_temp_4 * b_temp_1;
                acc_51 += a_temp_5 * b_temp_1;

                acc_42 += a_temp_4 * b_temp_2;
                acc_52 += a_temp_5 * b_temp_2;
              }

              double_v res_11 =
                  double_v(&C_padded[C_base_index + (x + 0) * L1_Y + y]);
              res_11 += acc_11;
              res_11.store(&C_padded[C_base_index + (x + 0) * L1_Y + y]);
              double_v res_21 =
                  double_v(&C_padded[C_base_index + (x + 1) * L1_Y + y]);
              res_21 += acc_21;
              res_21.store(&C_padded[C_base_index + (x + 1) * L1_Y + y]);
              double_v res_31 =
                  double_v(&C_padded[C_base_index + (x + 2) * L1_Y + y]);
              res_31 += acc_31;
              res_31.store(&C_padded[C_base_index + (x + 2) * L1_Y + y]);
              double_v res_41 =
                  double_v(&C_padded[C_base_index + (x + 3) * L1_Y + y]);
              res_41 += acc_41;
              res_41.store(&C_padded[C_base_index + (x + 3) * L1_Y + y]);

              double_v res_51 =
                  double_v(&C_padded[C_base_index + (x + 4) * L1_Y + y]);
              res_51 += acc_51;
              res_51.store(&C_padded[C_base_index + (x + 4) * L1_Y + y]);

              double_v res_12 =
                  double_v(&C_padded[C_base_index + (x + 0) * L1_Y + (y + 4)]);
              res_12 += acc_12;
              res_12.store(&C_padded[C_base_index + (x + 0) * L1_Y + (y + 4)]);
              double_v res_22 =
                  double_v(&C_padded[C_base_index + (x + 1) * L1_Y + (y + 4)]);
              res_22 += acc_22;
              res_22.store(&C_padded[C_base_index + (x + 1) * L1_Y + (y + 4)]);
              double_v res_32 =
                  double_v(&C_padded[C_base_index + (x + 2) * L1_Y + (y + 4)]);
              res_32 += acc_32;
              res_32.store(&C_padded[C_base_index + (x + 2) * L1_Y + (y + 4)]);
              double_v res_42 =
                  double_v(&C_padded[C_base_index + (x + 3) * L1_Y + (y + 4)]);
              res_42 += acc_42;
              res_42.store(&C_padded[C_base_index + (x + 3) * L1_Y + (y + 4)]);

              double_v res_52 =
                  double_v(&C_padded[C_base_index + (x + 4) * L1_Y + (y + 4)]);
              res_52 += acc_52;
              res_52.store(&C_padded[C_base_index + (x + 4) * L1_Y + (y + 4)]);
            }
          }

        });
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double>(end - start).count();
  }

  std::cout << "duration inner: " << duration << "s" << std::endl;

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

  // std::vector<double> C_return(N * N);
  // iterate_indices<2>(pol_copy, { 0, 0 }, { N, N },
  // 		       [this, &C_return, &C_conflict, N_fixed](size_t x, size_t y)
  // {
  // 			 C_return[x * N + y] = C_conflict[x * N_fixed + y];
  // 		       });

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
