// this also includes hpx headers, have to be included first
#include "index_iterator.hpp"

#include <boost/align/aligned_allocator.hpp>

#include <Vc/Vc>
using Vc::double_v;

#include "../util/util.hpp"

#include "opttmp/loop/unroll_loop.hpp"
#include "opttmp/memory_layout/tile_array.hpp"
#include "parameters.hpp"
#include <chrono>
#include <vector>

// not tuned for now
// #define L3_X 420 // max 2 L3 par set to 1024 (rest 512)
// #define L3_Y 256
// #define L3_K_STEP 256

// #define Y_REG 8
constexpr size_t X_REG = 5;                    // cannot be changed!
constexpr size_t Y_REG = 2 * double_v::size(); // cannot be changed!

using namespace index_iterator;

extern "C" bool is_valid_parameter_combination() {
  if (L1_X < X_REG) {
    std::cout << "error: L1_X < X_REG, L1_X too small" << std::endl;
    return false;
  }
  if (L1_Y < Y_REG) {
    std::cout << "error: L1_Y < Y_REG, L1_Y too small" << std::endl;
    return false;
  }
  if (!((L2_X % L1_X == 0) && (L3_X % L2_X == 0))) {
    std::cout << "error: x direction blocking not set up correctly"
              << std::endl;
    return false;
  }
  if (!((L2_Y % L1_Y == 0) && (L3_Y % L2_Y == 0))) {
    std::cout << "error: y direction blocking not set up correctly"
              << std::endl;
    return false;
  }
  if (!((L2_K_STEP % L1_K_STEP == 0) && (L3_K_STEP % L2_K_STEP == 0))) {
    std::cout << "error: k direction blocking not set up correctly"
              << std::endl;
    return false;
  }
  return true;
}

extern "C" std::vector<double>
combined_kernel(std::size_t N_org, std::size_t X_size, std::size_t Y_size,
                std::size_t K_size, std::vector<double> &A,
                std::vector<double> &B, size_t repetitions, double &duration) {

  std::vector<double, boost::alignment::aligned_allocator<double, 32>> C_padded(
      X_size * Y_size);

  // // create a matrix of l1 cachable submatrices, caching by tiling, no large
  // // strides even without padding
  // // is also padded if padding is enabled (row padded only)
  // std::vector<double, boost::alignment::aligned_allocator<double, 32>>
  // A_trans(
  //     K_size * X_size);
  // for (size_t l1_x = 0; l1_x < X_size / L1_X; l1_x += 1) {
  //   for (size_t l1_k = 0; l1_k < K_size / L1_K_STEP; l1_k += 1) {
  //     // look up submatrix
  //     size_t base_index = (L1_X * L1_K_STEP) * (l1_k * (X_size / L1_X) +
  //     l1_x);
  //     for (size_t x = 0; x < L1_X; x++) {
  //       for (size_t k = 0; k < L1_K_STEP; k++) {
  //         A_trans[base_index + k * L1_X + x] =
  //             A[(l1_x * L1_X + x) * K_size + (l1_k * L1_K_STEP + k)];
  //       }
  //     }
  //   }
  // }

  std::vector<memory_layout::tiling_info_dim> tiling_spec_A_trans(2);
  // tiling_spec[0].tile_size_dir = L1_X;
  // tiling_spec[0].stride = X_size;
  // tiling_spec[1].tile_size_dir = L1_K_STEP;
  // tiling_spec[1].stride = K_size;
  tiling_spec_A_trans[0].tile_size_dir = L1_K_STEP;
  tiling_spec_A_trans[0].stride = K_size;
  tiling_spec_A_trans[1].tile_size_dir = L1_X;
  tiling_spec_A_trans[1].stride = X_size;

  std::vector<double, boost::alignment::aligned_allocator<double, 32>>
      A_trans_untiled(A.size());
  for (size_t i = 0; i < K_size; i++) {
    for (size_t j = 0; j < X_size; j++) {
      A_trans_untiled[i * X_size + j] = A[j * K_size + i];
    }
  }

  // std::cout << "A_trans_untiled:" << std::endl;
  // print_matrix_host(K_size, X_size, A_trans_untiled);

  std::vector<double, boost::alignment::aligned_allocator<double, 32>> A_trans =
      memory_layout::make_tiled<2>(A_trans_untiled, tiling_spec_A_trans);

  // std::cout << "A_trans (tiled):" << std::endl;
  // print_matrix_host(K_size, X_size, A_trans);

  // // don't need padding for B, no dependency to row count
  // std::vector<double, boost::alignment::aligned_allocator<double, 32>>
  // B_padded(
  //     K_size * Y_size);
  // for (size_t l1_y = 0; l1_y < (Y_size / L1_Y); l1_y += 1) {
  //   for (size_t l1_k = 0; l1_k < (K_size / L1_K_STEP); l1_k += 1) {
  //     size_t base_index = (L1_Y * L1_K_STEP) *
  //                         (l1_k * (Y_size / L1_Y) + l1_y); // look up
  //                         submatrix
  //     for (size_t y = 0; y < L1_Y; y++) {
  //       for (size_t k = 0; k < L1_K_STEP; k++) {
  //         B_padded[base_index + k * L1_Y + y] =
  //             B[(l1_k * L1_K_STEP + k) * Y_size + (l1_y * L1_Y + y)];
  //       }
  //     }
  //   }
  // }

  std::vector<memory_layout::tiling_info_dim> tiling_spec_B(2);
  // tiling_spec[0].tile_size_dir = L1_X;
  // tiling_spec[0].stride = X_size;
  // tiling_spec[1].tile_size_dir = L1_K_STEP;
  // tiling_spec[1].stride = K_size;
  tiling_spec_B[0].tile_size_dir = L1_K_STEP;
  tiling_spec_B[0].stride = K_size;
  tiling_spec_B[1].tile_size_dir = L1_Y;
  tiling_spec_B[1].stride = Y_size;

  // because of allocator
  std::vector<double, boost::alignment::aligned_allocator<double, 32>> B_copy(
      B.size());
  std::copy(B.begin(), B.end(), B_copy.begin());

  std::vector<double, boost::alignment::aligned_allocator<double, 32>>
      B_padded = memory_layout::make_tiled<2>(B_copy, tiling_spec_B);

  // std::cout << "B_padded (tiled):" << std::endl;
  // print_matrix_host(K_size, Y_size, B_padded);

  // std::vector<size_t> min = {0, 0, 0};
  // std::vector<size_t> max = {X_size, Y_size, K_size};

  for (size_t rep = 0; rep < repetitions; rep++) {
    // reset result before every iteration
    // because C is zero-initialized, no explicit tiling step is required
    std::fill(C_padded.begin(), C_padded.end(), 0.0);

    // blocking_pseudo_execution_policy<size_t> policy(3);
    // // specify with ascending cache level
    // policy.set_final_steps({L1_X, L1_Y, L1_K_STEP});
    // policy.add_blocking({L2_X, L2_Y, L2_K_STEP}, {false, false, false});
    // policy.add_blocking({L3_X, L3_Y, L3_K_STEP},
    //                     {true, true, false}); // LLC blocking

    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();

// bool first = true;

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
                  for (size_t l1_y = l2_y; l1_y < l2_y + L2_Y; l1_y += L1_Y) {
                    for (size_t l1_k = l2_k; l1_k < l2_k + L2_K_STEP;
                         l1_k += L1_K_STEP) {

                      // std::vector<size_t> min = {l3_x, l3_y, l3_k};
                      // std::vector<size_t> max = {l3_x + L3_X, l3_y + L3_Y,
                      //                            l3_k + L3_K_STEP};

                      // iterate_indices<3>(policy, min, max, [&first,
                      // &C_padded,
                      //                                       &A_trans,
                      //                                       &B_padded,
                      //                                       X_size, Y_size](
                      //                                          size_t l1_x,
                      //                                          size_t l1_y,
                      //                                          size_t l1_k) {
                      size_t l1_block_x = l1_x / L1_X;
                      size_t l1_block_y = l1_y / L1_Y;
                      size_t C_base_index =
                          (L1_X * L1_Y) *
                          (l1_block_x * (Y_size / L1_Y) + l1_block_y);
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

                          for (size_t k_inner = 0; k_inner < L1_K_STEP;
                               k_inner += 1) {

                            double_v b_temp_1 = double_v(
                                &B_padded[B_base_index + k_inner * L1_Y + y],
                                Vc::flags::vector_aligned);
                            double_v b_temp_2 = double_v(
                                &B_padded[B_base_index + k_inner * L1_Y +
                                          (y + double_v::size())], // 4
                                Vc::flags::vector_aligned);

                            // loads from A_trans are broadcasts!
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

                          double_v res_51 = double_v(
                              &C_padded[C_base_index + (x + 4) * L1_Y + y],
                              Vc::flags::element_aligned);
                          res_51 += acc_51;
                          res_51.memstore(
                              &C_padded[C_base_index + (x + 4) * L1_Y + y],
                              Vc::flags::element_aligned);

                          double_v res_12 =
                              double_v(&C_padded[C_base_index + (x + 0) * L1_Y +
                                                 (y + double_v::size())], // 4
                                       Vc::flags::element_aligned);
                          res_12 += acc_12;
                          res_12.memstore(
                              &C_padded[C_base_index + (x + 0) * L1_Y +
                                        (y + double_v::size())], // 4
                              Vc::flags::element_aligned);
                          double_v res_22 =
                              double_v(&C_padded[C_base_index + (x + 1) * L1_Y +
                                                 (y + double_v::size())], // 4
                                       Vc::flags::element_aligned);
                          res_22 += acc_22;
                          res_22.memstore(
                              &C_padded[C_base_index + (x + 1) * L1_Y +
                                        (y + double_v::size())], // 4
                              Vc::flags::element_aligned);
                          double_v res_32 =
                              double_v(&C_padded[C_base_index + (x + 2) * L1_Y +
                                                 (y + double_v::size())], // 4
                                       Vc::flags::element_aligned);
                          res_32 += acc_32;
                          res_32.memstore(
                              &C_padded[C_base_index + (x + 2) * L1_Y +
                                        (y + double_v::size())], // 4
                              Vc::flags::element_aligned);
                          double_v res_42 =
                              double_v(&C_padded[C_base_index + (x + 3) * L1_Y +
                                                 (y + double_v::size())], // 4
                                       Vc::flags::element_aligned);
                          res_42 += acc_42;
                          res_42.memstore(
                              &C_padded[C_base_index + (x + 3) * L1_Y +
                                        (y + double_v::size())], // 4
                              Vc::flags::element_aligned);

                          double_v res_52 =
                              double_v(&C_padded[C_base_index + (x + 4) * L1_Y +
                                                 (y + double_v::size())], // 4
                                       Vc::flags::element_aligned);
                          res_52 += acc_52;
                          res_52.memstore(
                              &C_padded[C_base_index + (x + 4) * L1_Y +
                                        (y + double_v::size())], // 4
                              Vc::flags::element_aligned);
                        }
                      }

                      // if (first) {
                      //   first = false;
                      // }

                      // });
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

  std::vector<memory_layout::tiling_info_dim> tiling_spec_C(2);
  // tiling_spec[0].tile_size_dir = L1_X;
  // tiling_spec[0].stride = X_size;
  // tiling_spec[1].tile_size_dir = L1_K_STEP;
  // tiling_spec[1].stride = K_size;
  // tiling_spec_C[0].tile_size_dir = L1_X;
  // tiling_spec_C[0].stride = X_size;
  // tiling_spec_C[1].tile_size_dir = L1_Y;
  // tiling_spec_C[1].stride = Y_size;
  tiling_spec_C[0].tile_size_dir = L1_Y;
  tiling_spec_C[0].stride = Y_size;
  tiling_spec_C[1].tile_size_dir = L1_X;
  tiling_spec_C[1].stride = X_size;

  std::cout << "X_size: " << X_size << std::endl;
  std::cout << "Y_size: " << Y_size << std::endl;
  std::cout << "C_padded (tiled):" << std::endl;
  print_matrix_host(Y_size, X_size, C_padded);

  std::cout << "before undo_tiling" << std::endl;

  std::vector<double, boost::alignment::aligned_allocator<double, 32>>
      C_untiled = memory_layout::undo_tiling<2>(C_padded, tiling_spec_C);

  std::cout << "after undo_tiling" << std::endl;

  // std::vector<double> C_return(N_org * N_org);
  // std::copy(C_untiled.begin(), C_untiled.end(), C_return.begin());

  std::cout << "C_untiled:" << std::endl;
  print_matrix_host(Y_size, X_size, C_untiled);

  std::vector<double> C_return(N_org * N_org);
  for (size_t x = 0; x < N_org; x++) {
    for (size_t y = 0; y < N_org; y++) {
      C_return.at(y * N_org + x) = C_untiled.at(y * X_size + x);
    }
  }

  // std::cout << "C_return:" << std::endl;
  // print_matrix_host(N_org, C_return);

  // std::fill(C_return.begin(), C_return.end(), 0.0);

  // for (size_t l1_x = 0; l1_x < X_size / L1_X; l1_x += 1) {
  //   for (size_t l1_y = 0; l1_y < Y_size / L1_Y; l1_y += 1) {
  //     // look up submatrix
  //     size_t base_index =
  //         (L1_X * L1_Y) * (l1_x * (Y_size / L1_Y) + l1_y);
  //     for (size_t x = 0; x < L1_X; x++) {
  //       for (size_t y = 0; y < L1_Y; y++) {
  //         // skip padding
  //         if (l1_x * L1_X + x < N_org && l1_y * L1_Y + y < N_org) {
  //           C_return.at((l1_x * L1_X + x) * N_org + (l1_y * L1_Y + y)) =
  //               C_padded.at(base_index + x * L1_Y + y);
  //         }
  //       }
  //     }
  //   }
  // }
  return C_return;
}
