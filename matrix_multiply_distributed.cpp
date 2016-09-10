/*
 * matrix_multiplication_component.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: pfandedd
 */

#include <iostream>
#include <hpx/include/lcos.hpp>
#include <hpx/include/iostreams.hpp>
#include "matrix_multiply_distributed.hpp"

HPX_REGISTER_COMPONENT(hpx::components::component<matrix_multiply_distributed>,
		matrix_multiply_distributed);

HPX_REGISTER_ACTION(matrix_multiply_distributed::matrix_multiply_action);

std::vector<double> matrix_multiply_distributed::matrix_multiply(std::uint64_t x,
		std::uint64_t y, size_t blockSize) {
	C.resize(blockSize * blockSize);
	hpx::cout << "hi from node: " << hpx::find_here() << std::endl << hpx::flush;
	return C;
}
