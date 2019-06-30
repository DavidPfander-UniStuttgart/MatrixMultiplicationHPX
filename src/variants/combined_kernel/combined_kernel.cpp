// this also includes hpx headers, have to be included first
#include "index_iterator.hpp"

#include <boost/align/aligned_allocator.hpp>

#include <Vc/Vc>
using Vc::double_v;

#include "../util/util.hpp"

#include "opttmp/loop/unroll_loop.hpp"
#include "opttmp/memory_layout/tile_array.hpp"
#include "parameters.hpp"
#include <autotune/queue_thread_pool.hpp>
#include <chrono>
#include <opttmp/memory_layout/tile_view.hpp>
#include <opttmp/vectorization/register_tiling.hpp>
#include <vector>

#include <numa.h>
#include <sched.h>
#include <omp.h>

constexpr size_t Y_REG = Y_BASE_WIDTH * double_v::size(); // don't set directly
using reg_array = opttmp::vectorization::register_array<double_v, Y_BASE_WIDTH>;

/*#define L1_X (X_REG*L1_X_FACTOR)
#define L1_Y (Y_REG*L1_Y_FACTOR)
#define L2_X (X_REG*L1_X_FACTOR*L2_X_FACTOR)
#define L2_Y (Y_REG*L1_Y_FACTOR*L2_Y_FACTOR)
#define L2_K_STEP (L1_K_STEP*L2_K_FACTOR)*/

using namespace index_iterator;

extern "C" bool is_valid_parameter_combination() {

  if (L1_X < X_REG) {
    std::cout << "error: L1_X < X_REG, L1_X too small" << std::endl;
    return false;
  }
  if (L2_X < L1_X) {
    std::cout << "error: L2_X < L1_X, L2_X too small" << std::endl;
    return false;
  }
  if (L1_Y < Y_REG) {
    std::cout << "error: L1_Y < Y_REG, L1_Y too small" << std::endl;
    return false;
  }
  if (L2_Y < L1_Y) {
    std::cout << "error: L2_Y < L1_Y, L2_Y too small" << std::endl;
    return false;
  }
  if (L2_K_STEP < L1_K_STEP) {
    std::cout << "error: L2_K_STEP < L1_K_STEP, L2_K_STEP too small"
              << std::endl;
    return false;
  }
  if (L1_X % X_REG != 0) {
    std::cout << "error: L1_X does not divide X_REG" << std::endl;
    return false;
  }
  if (L1_Y % Y_REG != 0) {
    std::cout << "error: L1_Y does not divide Y_REG" << std::endl;
    return false;
  }
  if (L2_X % L1_X != 0) {
    std::cout << "error: x direction blocking error: L2_X % L1_X != 0"
              << std::endl;
    return false;
  }
  if (L2_Y % L1_Y != 0) {
    std::cout << "error: y direction blocking error: L2_Y % L1_Y != 0"
              << std::endl;
    return false;
  }
  if (L2_K_STEP % L1_K_STEP != 0) {
    std::cout
        << "error: k direction blocking error: L2_K_STEP % L1_K_STEP != 0 "
        << std::endl;
    return false;
  }
  /*if (L1_X > 40) {
    std::cout << "error: L1_X to large" << std::endl;
    return false;
  }
  if (L1_Y > 64) {
    std::cout << "error: L1_Y to large" << std::endl;
    return false;
  }
  if (L2_X > 100) {
    std::cout << "error: L2_X to large" << std::endl;
    return false;
  }
  if (L2_Y > 128) {
    std::cout << "error: L2_Y to large" << std::endl;
    return false;
  }
  if (L2_K_STEP > 256) {
    std::cout << "error: L2_K_STEP to large" << std::endl;
    return false;
  }*/
  return true;
}

void pad_matrices(const size_t N_org, const std::vector<double> &A_org,
                  const std::vector<double> &B_org, size_t &X_size,
                  size_t &Y_size, size_t &K_size, std::vector<double> &A_padded,
                  std::vector<double> &B_padded) {
  // k direction padding
  size_t k_pad = L2_K_STEP - (N_org % L2_K_STEP);
  if (k_pad == L2_K_STEP) {
    k_pad = 0; // nothing to pad
  }
  size_t x_pad = L2_X - (N_org % L2_X);
  if (x_pad == L2_X) {
    x_pad = 0; // nothing to pad
  }
  size_t y_pad = L2_Y - (N_org % L2_Y);
  if (y_pad == L2_Y) {
    y_pad = 0; // nothing to pad
  }

  std::cout << "matrix padding: x_pad = " << x_pad << ", y_pad = " << y_pad
            << ", k_pad = " << k_pad << std::endl;

  X_size = N_org + x_pad;
  Y_size = N_org + y_pad;
  K_size = N_org + k_pad;

  std::cout << "matrix dimensions for calculation: X = " << X_size
            << ", Y = " << Y_size << ", K = " << K_size << std::endl;

  A_padded = std::vector<double>(X_size * K_size);
  std::fill(A_padded.begin(), A_padded.end(), 0.0);
  for (size_t x = 0; x < N_org; x++) {
    for (size_t k = 0; k < N_org; k++) {
      A_padded.at(x * K_size + k) = A_org.at(x * N_org + k);
    }
  }
  B_padded = std::vector<double>(K_size * Y_size);
  std::fill(B_padded.begin(), B_padded.end(), 0.0);
  for (size_t y = 0; y < N_org; y++) {
    for (size_t k = 0; k < N_org; k++) {
      B_padded.at(k * Y_size + y) = B_org.at(k * N_org + y);
    }
  }
}

extern "C" std::vector<double> combined_kernel(std::size_t N_org,
                                               std::vector<double> &A_org,
                                               std::vector<double> &B_org,
                                               size_t repetitions,
                                               double &duration) {

  int num_nodes = numa_num_configured_nodes();
  std::cout << "num_nodes: " << num_nodes << std::endl;
  int num_cpus = numa_num_configured_cpus();
  std::cout << "num_cpus: " << num_cpus << std::endl;
  int cpus_per_node = num_cpus / num_nodes;
  std::cout << "cpus_per_node: " << cpus_per_node << std::endl;

  duration = 0.0;

  std::size_t X_size;
  std::size_t Y_size;
  std::size_t K_size;
  std::vector<double> A; // padded
  std::vector<double> B; // padded

  pad_matrices(N_org, A_org, B_org, X_size, Y_size, K_size, A, B);

  std::vector<double, boost::alignment::aligned_allocator<double, 64>> C_padded(
      X_size * Y_size);

  memory_layout::tiling_configuration tiling_spec_A_trans(2);
  tiling_spec_A_trans[0].tile_size_dir = L1_K_STEP;
  tiling_spec_A_trans[0].stride = K_size;
  tiling_spec_A_trans[1].tile_size_dir = L1_X;
  tiling_spec_A_trans[1].stride = X_size;

  std::vector<double, boost::alignment::aligned_allocator<double, 64>>
      A_trans_untiled(A.size());
  for (size_t i = 0; i < K_size; i++) {
    for (size_t j = 0; j < X_size; j++) {
      A_trans_untiled[i * X_size + j] = A[j * K_size + i];
    }
  }

// std::cout << "A_trans_untiled:" << std::endl;
// print_matrix_host(K_size, X_size, A_trans_untiled);

#if KERNEL_NUMA == 1
  std::vector<double, boost::alignment::aligned_allocator<double, 64>>
      A_trans_some_node =
          memory_layout::make_tiled<2>(A_trans_untiled, tiling_spec_A_trans);
#else
  std::vector<double, boost::alignment::aligned_allocator<double, 64>> A_trans =
      memory_layout::make_tiled<2>(A_trans_untiled, tiling_spec_A_trans);
#endif

  // std::cout << "A_trans (tiled):" << std::endl;
  // print_matrix_host(K_size, X_size, A_trans);

  memory_layout::tiling_configuration tiling_spec_B(2);
  tiling_spec_B[0].tile_size_dir = L1_K_STEP;
  tiling_spec_B[0].stride = K_size;
  tiling_spec_B[1].tile_size_dir = L1_Y;
  tiling_spec_B[1].stride = Y_size;

  // because of allocator
  std::vector<double, boost::alignment::aligned_allocator<double, 64>> B_copy(
      B.size());
  std::copy(B.begin(), B.end(), B_copy.begin());

#if KERNEL_NUMA == 1
  std::vector<double, boost::alignment::aligned_allocator<double, 64>>
      B_padded_some_node = memory_layout::make_tiled<2>(B_copy, tiling_spec_B);

  std::vector<
      std::vector<double, boost::alignment::aligned_allocator<double, 64>>>
      A_trans_nodes(num_nodes);
  for (int i = 0; i < num_nodes; i += 1) {
    numa_run_on_node(i);
    A_trans_nodes[i] = A_trans_some_node;
  }

  std::vector<
      std::vector<double, boost::alignment::aligned_allocator<double, 64>>>
      B_padded_nodes(num_nodes);
  for (int i = 0; i < num_nodes; i += 1) {
    numa_run_on_node(i);
    B_padded_nodes[i] = B_padded_some_node;
  }
#else
  std::vector<double, boost::alignment::aligned_allocator<double, 64>>
      B_padded = memory_layout::make_tiled<2>(B_copy, tiling_spec_B);
#endif

  memory_layout::tiling_configuration tiling_spec_C(2);
  tiling_spec_C[0].tile_size_dir = L1_X;
  tiling_spec_C[0].stride = X_size;
  tiling_spec_C[1].tile_size_dir = L1_Y;
  tiling_spec_C[1].stride = Y_size;

  for (size_t rep = 0; rep < repetitions; rep++) {
    // reset result before every iteration
    // because C is zero-initialized, no explicit tiling step is required
    std::fill(C_padded.begin(), C_padded.end(), 0.0);

    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();

#define use_omp
#ifdef use_omp
#if KERNEL_SCHEDULE == 1
#pragma omp parallel for collapse(2), num_threads(KERNEL_OMP_THREADS), schedule(dynamic)
#else
#pragma omp parallel for collapse(2), num_threads(KERNEL_OMP_THREADS), schedule(static)
#endif
#else
    autotune::queue_thread_pool<KERNEL_OMP_THREADS> pool;
    // pool.set_affinity(autotune::affinity_type_t::compact);
    pool.start();
#endif
    for (size_t l2_x = 0; l2_x < X_size; l2_x += L2_X) {
      for (size_t l2_y = 0; l2_y < Y_size; l2_y += L2_Y) {
#ifndef use_omp	  
        pool.enqueue_work(
            [&](size_t l2_x, size_t l2_y) -> void {
#endif
// get pointers to the matrices on your numa node
#if KERNEL_NUMA == 1
#ifdef use_omp
              int thread_id = omp_get_thread_num();
#else
              int thread_id = sched_getcpu();
#endif
              int mapped_numa_node = numa_node_of_cpu(thread_id);
              numa_run_on_node(mapped_numa_node);
              // std::cout << "thread_id: " << thread_id
              //           << " my numa node is: " << mapped_numa_node
              //           << std::endl;
              std::vector<double,
                          boost::alignment::aligned_allocator<double, 64>>
                  &A_trans = A_trans_nodes[mapped_numa_node];
              std::vector<double,
                          boost::alignment::aligned_allocator<double, 64>>
                  &B_padded = B_padded_nodes[mapped_numa_node];
#endif

              for (size_t l2_k = 0; l2_k < K_size; l2_k += L2_K_STEP) {
                // L1 blocking
                for (size_t l1_x = l2_x;
                     (l1_x < l2_x + L2_X) && (l1_x < X_size); l1_x += L1_X) {
                  for (size_t l1_y = l2_y;
                       (l1_y < l2_y + L2_Y) && (l1_y < Y_size); l1_y += L1_Y) {
                    auto C_view = memory_layout::make_view_from_index<2>(
                        {l1_x, l1_y}, C_padded, tiling_spec_C);
                    for (size_t l1_k = l2_k;
                         (l1_k < l2_k + L2_K_STEP) && (l1_k < K_size);
                         l1_k += L1_K_STEP) {
                      auto A_trans_view =
                          memory_layout::make_view_from_index<2>(
                              {l1_k, l1_x}, A_trans, tiling_spec_A_trans);
                      auto B_view = memory_layout::make_view_from_index<2>(
                          {l1_k, l1_y}, B_padded, tiling_spec_B);

                      // Register blocking
                      for (size_t x = 0; x < L1_X; x += X_REG) {
                        for (size_t y = 0; y < L1_Y; y += Y_REG) {

                          std::array<reg_array, X_REG> acc;

                          for (size_t k_inner = 0; k_inner < L1_K_STEP;
                               k_inner += 1) {
                            const reg_array b_temp(
                                B_view.pointer(k_inner * L1_Y + y),
                                Vc::flags::vector_aligned);

                            // loads from A_trans are broadcasts!
                            std::array<double_v, X_REG> a_temp;
                            for (size_t r = 0; r < X_REG; r++) {
                              a_temp[r] =
                                  A_trans_view[k_inner * L1_X + (x + r)];
                            }

                            for (size_t r = 0; r < X_REG; r++) {
                              acc[r] += a_temp[r] * b_temp;
                            }
                          }

                          for (size_t r = 0; r < X_REG; r++) {
                            double *const res_ptr =
                                C_view.pointer((x + r) * L1_Y + y);
                            reg_array res_value(res_ptr,
                                                Vc::flags::element_aligned);
                            res_value += acc[r];
                            res_value.memstore(res_ptr,
                                               Vc::flags::element_aligned);
                          } // X_REG
                        }   // Y_REG
                      }     // L1_X
                            // #endif
                    }       // L1_K_STEP
                  }         // L1_Y
                }           // L1_Y
              }
#ifndef use_omp
            },
            l2_x, l2_y); // L2_K_STEP
#endif
      }                  // L2_Y
    }                    // L2_X

#ifndef use_omp
    pool.finish();
#endif

    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double>(end - start).count();
  }

  std::vector<double, boost::alignment::aligned_allocator<double, 64>>
      C_untiled = memory_layout::undo_tiling<2>(C_padded, tiling_spec_C);

  std::vector<double> C_return(N_org * N_org);
  for (size_t r = 0; r < N_org; r++) {
    for (size_t c = 0; c < N_org; c++) {
      C_return.at(r * N_org + c) = C_untiled.at(r * Y_size + c);
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
