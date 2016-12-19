#include "multiplier.hpp"

#include "../../reference_kernels/kernel.hpp"
#include <hpx/include/iostreams.hpp>
#include <hpx/include/lcos.hpp>
#include <iostream>

HPX_REGISTER_COMPONENT(
    hpx::components::component<multiply_components::multiplier>, multiplier);

HPX_REGISTER_ACTION(
    multiply_components::multiplier::calculate_submatrix_action);

namespace multiply_components {

std::vector<double> multiplier::calculate_submatrix(std::uint64_t x,
                                                    std::uint64_t y,
                                                    size_t block_result) {
  // hpx::cout << "block_result: " << block_result << std::endl << hpx::flush;
  std::vector<double> C = std::vector<double>(block_result * block_result,
                                              0.0); // initialize to zero

  if (!transposed) {
    kernel::kernel(A, B, C, N, x, y, block_result);
  } else {
    // no blocking?
    if (block_input == 0 || block_result < 4) {
      kernel::kernel_transposed(A, B, C, N, x, y, block_result);
    } else {
      kernel::kernel_transposed_blocked(A, B, C, N, x, y, block_result,
                                        block_input);
    }
  }
  return C;
}
}
