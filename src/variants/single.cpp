#include "single.hpp"

#include "variants/components/multiplier.hpp"
#include "variants/components/recursive.hpp"

namespace single {

single::single(size_t N, std::vector<double> &A, std::vector<double> &B,
               bool transposed, uint64_t block_result, uint64_t block_input,
               uint64_t repetitions, uint64_t verbose)
    : N(N), A(A), B(B), transposed(transposed), block_result(block_result),
      block_input(block_input), repetitions(repetitions), verbose(verbose) {}

std::vector<double> single::matrix_multiply() {

  std::vector<double> C(N * N);

  hpx::cout << "using parallel single node algorithm" << std::endl
            << hpx::flush;
  hpx::cout << "warning: stack overflows can occur depending on block_input, "
               "block_result and the size of the matrix"
            << std::endl
            << hpx::flush;
  hpx::components::client<multiply_components::multiplier> multiplier =
      hpx::new_<hpx::components::client<multiply_components::multiplier>>(
          hpx::find_here(), N, A, B, transposed, block_input, verbose);
  uint32_t comp_locality_multiplier =
      hpx::naming::get_locality_id_from_id(multiplier.get_id());
  multiplier.register_as(
      "/multiplier#" + std::to_string(comp_locality_multiplier), false);
  hpx::components::client<multiply_components::recursive> recursive =
      hpx::new_<hpx::components::client<multiply_components::recursive>>(
          hpx::find_here(), block_result, verbose);
  uint32_t comp_locality_recursive =
      hpx::naming::get_locality_id_from_id(recursive.get_id());
  recursive.register_as("/recursive#" + std::to_string(comp_locality_recursive),
                        false);
  for (size_t repeat = 0; repeat < repetitions; repeat++) {
    auto f = hpx::async<
        multiply_components::recursive::distribute_recursively_action>(
        recursive.get_id(), 0, 0, N);
    C = f.get();
  }

  return C;
}
}
