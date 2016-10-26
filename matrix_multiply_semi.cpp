/*
 * matrix_multiply_algorithm.hpp
 *
 *  Created on: Oct 10, 2016
 *      Author: pfandedd
 */

#include "matrix_multiply_semi.hpp"

#include "index_iterator.hpp"

#include "hpx/parallel/algorithms/for_each.hpp"
#include "hpx/parallel/algorithms/for_loop.hpp"
#include "hpx/parallel/execution_policy.hpp"
#include <boost/iterator/iterator_facade.hpp>
#include "hpx/util/iterator_facade.hpp"
#include <hpx/include/iostreams.hpp>

using namespace index_iterator;

namespace semi {

  matrix_multiply_semi::matrix_multiply_semi(size_t N, std::vector<double> &A, std::vector<double> &B,
					     bool transposed, uint64_t block_result, uint64_t block_input,
					     uint64_t repetitions, uint64_t verbose) :
    N(N), A(A), B(B), transposed(transposed), block_result(block_result), block_input(
										      block_input), repetitions(repetitions), verbose(verbose) {

  }

  std::vector<double> matrix_multiply_semi::matrix_multiply() {

    // add single additional cacheline to avoid conflict misses
    // (cannot add only single data field, as then cache-boundaries are crossed)

    // N_fixed = N -> bad, conflict misses
    // N_fixed = N + 1 -> better, no conflict misses, but matrix rows not aligned to cache boundary (2 cache lines per load)
    // N_fixed = N + 4 -> bit better, for unknown reasons,
    // Same for N + 8, little bit better for N + 16, N + 32 even better, N + 64 same performance
    size_t N_fixed = N;

    std::vector<double> A_conflict(N_fixed * N_fixed);
    std::vector<double> B_conflict(N_fixed * N_fixed);

    blocking_pseudo_execution_policy<size_t> pol_copy(2);
    iterate_indices<2>(pol_copy, { 0, 0 }, { N, N },
		       [this, N_fixed, &A_conflict](size_t x, size_t y) {
			 A_conflict[x * N_fixed + y] = A[x * N + y];
		       });

    iterate_indices<2>(pol_copy, { 0, 0 }, { N, N },
		       [this, N_fixed, &B_conflict](size_t x, size_t y) {
			 B_conflict[x * N_fixed + y] = B[x * N + y];
		       });

    std::vector<double> C(N * N);
    std::fill(C.begin(), C.end(), 0.0);

    std::vector<double> C_conflict(N_fixed * N_fixed);
    std::fill(C_conflict.begin(), C_conflict.end(), 0.0);

    std::vector<size_t> min = { 0, 0, 0 };
    std::vector<size_t> max = { N, N, N };
    //        std::vector<size_t> block = { block_result, block_result, block_input };
    //        std::vector<bool> parallel_dims = { true, true, false };

    //        hpx::parallel::par;
    //        const hpx::parallel::parallel_execution_policy &t = hpx::parallel::seq;

    //        std::vector<
    //                std::reference_wrapper<
    //                        const hpx::parallel::parallel_execution_policy>>
    //        execution_policy =
    //        {   hpx::parallel::par, hpx::parallel::par, hpx::parallel::seq};

    blocking_pseudo_execution_policy<size_t> policy(3);
    policy.add_blocking({32, 32, 32}, {false, false, false}); // L1 blocking
           // policy.add_blocking({32, 32, 64}, {false, false, false}); // L1 blocking
    // policy.add_blocking( { block_result, block_result, block_input }, { false,
    // 	  false, false }); // L1 blocking
    //        policy.add_blocking({256, 256, 256}, {true, true, false}); // L1 blocking
    //        policy.add_blocking(block, parallel_dims); // L1 blocking
    //        policy.add_blocking({512, 512, 128}, {false, false, false}); // LLC blocking
    // policy.set_final_steps( { 4, 2, block_input });

    //        const size_t blocks_x = N / block_result; // N has to be divisible
    //        const size_t blocks_y = N / block_result; // N has to be divisible
    //        const size_t blocks_k = N / block_input; // N has to be divisible
    //        const size_t submatrix_size = block_result * block_input;

    //        compute_kernel_struct compute_kernel(A, B, C, N);
    //
    //        action_wrapper<size_t, compute_kernel_struct> wrap;
    hpx::cout << "now doing interesting stuff" << std::endl << hpx::flush;
    hpx::cout << "N: " << N << std::endl << hpx::flush;
hpx::cout << "------------before mat mul-----------------" << std::endl << hpx::flush;
    iterate_indices<3>(policy, min, max,
		       [N_fixed, &C_conflict, &A_conflict, &B_conflict, this](size_t x, size_t y, size_t k) {
			 hpx::cout << "x: " << x << " y: " << y << " k: " << k << std::endl << hpx::flush;

			 //                    hpx::cout << "x: " << x << " y: " << y << " k: " << k << std::endl << hpx::flush;
			 //                    C_conflict[x * N_fixed + y] += A_conflict[x * N_fixed + k] * B_conflict[y * N_fixed + k];

			 //                    auto A_submatrix_begin = A_conflict.begin() + x * N_fixed + k;
			 //                    auto A_submatrix_end = A_conflict.begin() + x * N_fixed + k + block_input;
			 //                    // assumes transposed input matrix
			 //                    auto B_submatrix_begin = B_conflict.begin() + y * N_fixed + k;
			 //
			 //                    // do auto-vectorized small matrix dot product
			 //                    C_conflict[x * N_fixed + y] += std::inner_product(A_submatrix_begin, A_submatrix_end, B_submatrix_begin, 0.0);

			 //                    // integer division with truncation!
			 //                    size_t block_x = x / block_result;
			 //                    size_t block_y = y / block_result;
			 //                    size_t block_k = k / 64; //k always divides blocks_k
			 //
			 //                    size_t inner_x = x % block_result;
			 //                    size_t inner_y = y % block_result;
			 //
			 //                    size_t offset_A = submatrix_size * (block_x * blocks_k + block_k);
			 //                    size_t offset_B = submatrix_size * (block_y * blocks_k + block_k);

			 double result_component_0_0 = 0.0;
			 double result_component_0_1 = 0.0;
			 double result_component_1_0 = 0.0;
			 double result_component_1_1 = 0.0;
			 double result_component_2_0 = 0.0;
			 double result_component_2_1 = 0.0;
			 double result_component_3_0 = 0.0;
			 double result_component_3_1 = 0.0;

			 for (size_t k_inner = 0; k_inner < block_input; k_inner++) {
			   //                        result_component_0_0 += A_conflict[offset_A + (inner_x + 0) * block_input + k_inner]
			   //                                * B_conflict[offset_B + (inner_y + 0) * block_input + k_inner];
			   //                        result_component_0_1 += A_conflict[offset_A + (inner_x + 0) * block_input + k_inner]
			   //                                * B_conflict[offset_B + (inner_y + 1) * block_input + k_inner];
			   //                        result_component_1_0 += A_conflict[offset_A + (inner_x + 1) * block_input + k_inner]
			   //                                * B_conflict[offset_B + (inner_y + 0) * block_input + k_inner];
			   //                        result_component_1_1 += A_conflict[offset_A + (inner_x + 1) * block_input + k_inner]
			   //                                * B_conflict[offset_B + (inner_y + 1) * block_input + k_inner];
			   //
			   //                        result_component_2_0 += A_conflict[offset_A + (inner_x + 2) * block_input + k_inner]
			   //                                * B_conflict[offset_B + (inner_y + 0) * block_input + k_inner];
			   //                        result_component_2_1 += A_conflict[offset_A + (inner_x + 2) * block_input + k_inner]
			   //                                * B_conflict[offset_B + (inner_y + 1) * block_input + k_inner];
			   //                        result_component_3_0 += A_conflict[offset_A + (inner_x + 3) * block_input + k_inner]
			   //                                * B_conflict[offset_B + (inner_y + 0) * block_input + k_inner];
			   //                        result_component_3_1 += A_conflict[offset_A + (inner_x + 3) * block_input + k_inner]
			   //                                * B_conflict[offset_B + (inner_y + 1) * block_input + k_inner];

			   result_component_0_0 += A_conflict[(x + 0) * N_fixed + k + k_inner]
			     * B_conflict[(y + 0) * N_fixed + k + k_inner];
			   result_component_0_1 += A_conflict[(x + 0) * N_fixed + k + k_inner]
			     * B_conflict[(y + 1) * N_fixed + k + k_inner];
			   result_component_1_0 += A_conflict[(x + 1) * N_fixed + k + k_inner]
			     * B_conflict[(y + 0) * N_fixed + k + k_inner];
			   result_component_1_1 += A_conflict[(x + 1) * N_fixed + k + k_inner]
			     * B_conflict[(y + 1) * N_fixed + k + k_inner];

			   result_component_2_0 += A_conflict[(x + 2) * N_fixed + k + k_inner]
			     * B_conflict[(y + 0) * N_fixed + k + k_inner];
			   result_component_2_1 += A_conflict[(x + 2) * N_fixed + k + k_inner]
			     * B_conflict[(y + 1) * N_fixed + k + k_inner];
			   result_component_3_0 += A_conflict[(x + 3) * N_fixed + k + k_inner]
			     * B_conflict[(y + 0) * N_fixed + k + k_inner];
			   result_component_3_1 += A_conflict[(x + 3) * N_fixed + k + k_inner]
			     * B_conflict[(y + 1) * N_fixed + k + k_inner];
			 }
			 // assumes matrix was zero-initialized
			 C_conflict[(x + 0) * N_fixed + (y + 0)] += result_component_0_0;
			 C_conflict[(x + 0) * N_fixed + (y + 1)] += result_component_0_1;
			 C_conflict[(x + 1) * N_fixed + (y + 0)] += result_component_1_0;
			 C_conflict[(x + 1) * N_fixed + (y + 1)] += result_component_1_1;

			 C_conflict[(x + 2) * N_fixed + (y + 0)] += result_component_2_0;
			 C_conflict[(x + 2) * N_fixed + (y + 1)] += result_component_2_1;
			 C_conflict[(x + 3) * N_fixed + (y + 0)] += result_component_3_0;
			 C_conflict[(x + 3) * N_fixed + (y + 1)] += result_component_3_1;
		       });

hpx::cout << "------------after mat mul-----------------" << std::endl << hpx::flush;

    iterate_indices<2>(pol_copy, { 0, 0 }, { N, N },
		       [this, &C, &C_conflict, N_fixed](size_t x, size_t y) {
			 C[x * N + y] = C_conflict[x * N_fixed + y];

		       });

    return C;
  }
}
