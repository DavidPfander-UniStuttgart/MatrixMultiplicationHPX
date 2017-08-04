// this also includes hpx headers, have to be included first
#include "index_iterator.hpp"

#include <boost/align/aligned_allocator.hpp>

#include <Vc/Vc>
using Vc::double_v;

#include "opttmp/loop/unroll_loop.hpp"
#include "parameters.hpp"
#include <vector>

#define X_REG 5 // cannot be changed!
#define Y_REG 8 // cannot be changed!

using namespace index_iterator;

extern "C" void combined_kernel(std::size_t N_org, std::size_t X_size,
                                std::size_t Y_size, std::size_t K_size,
                                std::vector<double> &A, std::vector<double> &B,
                                std::vector<double> &C_return, size_t repetitions,
                                double &duration) {

  // create a matrix of l1 cachable submatrices, caching by tiling, no large
  // strides even without padding
  std::vector<double, boost::alignment::aligned_allocator<double, 32>> C_padded(
      X_size * Y_size);
  std::fill(C_padded.begin(), C_padded.end(), 0.0);

  std::vector<double, boost::alignment::aligned_allocator<double, 32>>
      A_trans_untiled(K_size * X_size);
  for (size_t i = 0; i < K_size; i++) {
    for (size_t j = 0; j < X_size; j++) {
      A_trans_untiled[i * X_size + j] = A[j * K_size + i];
    }
  }

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

    bool first = true;

    iterate_indices<3>(policy, min, max, [&first, &C_padded, &A_trans,
                                          &B_padded, X_size,
                                          Y_size](size_t l1_x, size_t l1_y,
                                                  size_t l1_k) {
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

            double_v b_temp_1 =
                double_v(&B_padded[B_base_index + k_inner * L1_Y + y],
                         Vc::flags::vector_aligned);
            double_v b_temp_2 =
                double_v(&B_padded[B_base_index + k_inner * L1_Y + (y + 4)],
                         Vc::flags::vector_aligned);

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
              double_v(&C_padded[C_base_index + (x + 0) * L1_Y + y],
                       Vc::flags::element_aligned);
          res_11 += acc_11;
          res_11.memstore(&C_padded[C_base_index + (x + 0) * L1_Y + y],
                          Vc::flags::element_aligned);
          double_v res_21 =
              double_v(&C_padded[C_base_index + (x + 1) * L1_Y + y],
                       Vc::flags::element_aligned);
          res_21 += acc_21;
          res_21.memstore(&C_padded[C_base_index + (x + 1) * L1_Y + y],
                          Vc::flags::element_aligned);
          double_v res_31 =
              double_v(&C_padded[C_base_index + (x + 2) * L1_Y + y],
                       Vc::flags::element_aligned);
          res_31 += acc_31;
          res_31.memstore(&C_padded[C_base_index + (x + 2) * L1_Y + y],
                          Vc::flags::element_aligned);
          double_v res_41 =
              double_v(&C_padded[C_base_index + (x + 3) * L1_Y + y],
                       Vc::flags::element_aligned);
          res_41 += acc_41;
          res_41.memstore(&C_padded[C_base_index + (x + 3) * L1_Y + y],
                          Vc::flags::element_aligned);

          double_v res_51 =
              double_v(&C_padded[C_base_index + (x + 4) * L1_Y + y],
                       Vc::flags::element_aligned);
          res_51 += acc_51;
          res_51.memstore(&C_padded[C_base_index + (x + 4) * L1_Y + y],
                          Vc::flags::element_aligned);

          double_v res_12 =
              double_v(&C_padded[C_base_index + (x + 0) * L1_Y + (y + 4)],
                       Vc::flags::element_aligned);
          res_12 += acc_12;
          res_12.memstore(&C_padded[C_base_index + (x + 0) * L1_Y + (y + 4)],
                          Vc::flags::element_aligned);
          double_v res_22 =
              double_v(&C_padded[C_base_index + (x + 1) * L1_Y + (y + 4)],
                       Vc::flags::element_aligned);
          res_22 += acc_22;
          res_22.memstore(&C_padded[C_base_index + (x + 1) * L1_Y + (y + 4)],
                          Vc::flags::element_aligned);
          double_v res_32 =
              double_v(&C_padded[C_base_index + (x + 2) * L1_Y + (y + 4)],
                       Vc::flags::element_aligned);
          res_32 += acc_32;
          res_32.memstore(&C_padded[C_base_index + (x + 2) * L1_Y + (y + 4)],
                          Vc::flags::element_aligned);
          double_v res_42 =
              double_v(&C_padded[C_base_index + (x + 3) * L1_Y + (y + 4)],
                       Vc::flags::element_aligned);
          res_42 += acc_42;
          res_42.memstore(&C_padded[C_base_index + (x + 3) * L1_Y + (y + 4)],
                          Vc::flags::element_aligned);

          double_v res_52 =
              double_v(&C_padded[C_base_index + (x + 4) * L1_Y + (y + 4)],
                       Vc::flags::element_aligned);
          res_52 += acc_52;
          res_52.memstore(&C_padded[C_base_index + (x + 4) * L1_Y + (y + 4)],
                          Vc::flags::element_aligned);
        }
      }

      if (first) {
        first = false;
      }

    });

    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double>(end - start).count();
  }

  std::cout << "duration inner: " << duration << "s" << std::endl;

  // std::cout << "C_tiled or padded:" << std::endl;
  // print_matrix_host(Y_size, X_size, C_padded);

  // std::vector<double> C_return(N_org * N_org);
  // std::fill(C_return.begin(), C_return.end(), 0.0);

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
}
