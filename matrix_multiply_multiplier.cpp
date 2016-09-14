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

std::vector<double> matrix_multiply_multiplier::calculate_submatrix(std::uint64_t x, std::uint64_t y, size_t blockSize) {
  std::vector<double> C(blockSize * blockSize);
  kernel::matrix_multiply_kernel(A, B, C, N, x, y, blockSize);
  return C;
}
