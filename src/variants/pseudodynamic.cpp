#include "pseudodynamic.hpp"

#include "variants/static_improved.hpp"

namespace pseudodynamic {

pseudodynamic::pseudodynamic(size_t N, std::vector<double> &A,
                             std::vector<double> &B, bool transposed,
                             uint64_t block_result, uint64_t block_input,
                             size_t min_work_size, size_t max_work_difference,
                             double max_relative_work_difference,
                             uint64_t repetitions, uint64_t verbose)
    : N(N), A(A), B(B), transposed(transposed), block_result(block_result),
      block_input(block_input), min_work_size(min_work_size),
      max_work_difference(max_work_difference),
      max_relative_work_difference(max_relative_work_difference),
      repetitions(repetitions), verbose(verbose) {}

std::vector<double> pseudodynamic::matrix_multiply() {

	std::vector<double> C(N * N);

  multiply_components::static_improved m(
      N, A, B, transposed, block_input, block_result, min_work_size,
      max_work_difference, max_relative_work_difference, repetitions, verbose);
  C = m.matrix_multiply();

  return C;
}
}
