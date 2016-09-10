/*
 * matrix_multiplication_component.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: pfandedd
 */

#pragma once

#include <cinttypes>
#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>

extern uint64_t verbose;

struct matrix_multiply_distributed: hpx::components::component_base<
		matrix_multiply_distributed> {

	size_t N;
	std::vector<double> A;
	std::vector<double> B;
	std::vector<double> C;

	// TODO: why does this get called?
	matrix_multiply_distributed() :
			N(0) {
	}

	matrix_multiply_distributed(size_t N, std::vector<double> A,
			std::vector<double> B) :
			N(N), A(A), B(B) {
	}

	std::vector<double> matrix_multiply(std::uint64_t x, std::uint64_t y,
			size_t blockSize);

	HPX_DEFINE_COMPONENT_ACTION(matrix_multiply_distributed, matrix_multiply,
			matrix_multiply_action);

};

HPX_REGISTER_ACTION_DECLARATION(
		matrix_multiply_distributed::matrix_multiply_action);

