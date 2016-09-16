/*
 * matrix_multiplication_component.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: pfandedd
 */

#pragma once

#include <cinttypes>
#include <sstream>
#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>

struct matrix_multiply_multiplier: hpx::components::component_base<
		matrix_multiply_multiplier> {
	size_t N;
	std::vector<double> A;
	std::vector<double> B;
	bool transposed;
	// set to 0 to disable
	uint64_t block_input;
	uint64_t verbose;

	// TODO: why does this get called?
	matrix_multiply_multiplier() :
			N(0), transposed(false), block_input(0), verbose(0) {
	}

	matrix_multiply_multiplier(size_t N, std::vector<double> A,
			std::vector<double> B, bool transposed, uint64_t block_input,
			uint64_t verbose) :
	  N(N), A(std::move(A)), B(std::move(B)),
	  transposed(transposed), block_input(block_input), verbose(verbose)
  {
  }

	std::vector<double> calculate_submatrix(std::uint64_t x, std::uint64_t y,
			size_t block_result);

	HPX_DEFINE_COMPONENT_ACTION(matrix_multiply_multiplier, calculate_submatrix,
			calculate_submatrix_action);

};

HPX_REGISTER_ACTION_DECLARATION(
		matrix_multiply_multiplier::calculate_submatrix_action);

