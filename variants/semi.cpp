#include "semi.hpp"

#include "index_iterator.hpp"

#include "hpx/parallel/algorithms/for_each.hpp"
#include "hpx/parallel/algorithms/for_loop.hpp"
#include "hpx/parallel/execution_policy.hpp"
#include <boost/iterator/iterator_facade.hpp>
#include "hpx/util/iterator_facade.hpp"
#include <hpx/include/iostreams.hpp>

using namespace index_iterator;

namespace semi {

  semi::semi(size_t N, std::vector<double> &A, std::vector<double> &B,
					     bool transposed, uint64_t block_result, uint64_t block_input,
					     uint64_t repetitions, uint64_t verbose) :
    N(N), A(A), B(B), transposed(transposed), block_result(block_result), block_input(
										      block_input), repetitions(repetitions), verbose(verbose) {

  }

  std::vector<double> semi::matrix_multiply() {

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
    std::vector<size_t> block = { block_result, block_result, block_input };

    blocking_pseudo_execution_policy<size_t> policy(3);
    policy.add_blocking({4, 4, 4}, {true, true, false}); // L1 blocking
    policy.set_final_steps( { 4, 2, block_input });

    iterate_indices<3>(policy, min, max,
		       [N_fixed, &C_conflict, &A_conflict, &B_conflict, this](size_t x, size_t y, size_t k) {

			 double result_component_0_0 = 0.0;
			 double result_component_0_1 = 0.0;
			 double result_component_1_0 = 0.0;
			 double result_component_1_1 = 0.0;
			 double result_component_2_0 = 0.0;
			 double result_component_2_1 = 0.0;
			 double result_component_3_0 = 0.0;
			 double result_component_3_1 = 0.0;

			 for (size_t k_inner = 0; k_inner < block_input; k_inner++) {
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

    iterate_indices<2>(pol_copy, { 0, 0 }, { N, N },
		       [this, &C, &C_conflict, N_fixed](size_t x, size_t y) {
			 C[x * N + y] = C_conflict[x * N_fixed + y];

		       });

    return C;
  }
}
