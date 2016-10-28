/*
 * matrix_multiplication_component.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: pfandedd
 */

#include "matrix_multiply_multiplier.hpp"

#include <iostream>
#include <hpx/include/lcos.hpp>
#include <hpx/include/iostreams.hpp>
#include "matrix_multiply_kernel.hpp"

HPX_REGISTER_COMPONENT(hpx::components::component<matrix_multiply_multiplier>,
		       matrix_multiply_multiplier);

HPX_REGISTER_ACTION(matrix_multiply_multiplier::calculate_submatrix_action);

std::vector<double> matrix_multiply_multiplier::calculate_submatrix(
								    std::uint64_t x, std::uint64_t y, size_t block_result) {
  hpx::cout << "block_result: " << block_result << std::endl << hpx::flush;
  std::vector<double> C = std::vector<double>(block_result * block_result, 0.0); // initialize to zero
  if (!transposed) {
    kernel::matrix_multiply_kernel(A, B, C, N, x, y, block_result);
  } else {
    // no blocking?
    if (block_input == 0 || block_result < 4) {
      kernel::matrix_multiply_kernel_transposed(A, B, C, N, x, y,
						block_result);
    } else {
      kernel::matrix_multiply_kernel_transposed_blocked(A, B, C, N, x, y,
							block_result, block_input);
    }
  }
  return C;
}
