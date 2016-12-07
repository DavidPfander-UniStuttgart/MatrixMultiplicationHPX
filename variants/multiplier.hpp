#pragma once

#include <cinttypes>
#include <sstream>
#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>

namespace multiply_components {

struct multiplier: hpx::components::component_base<
		multiplier> {
	size_t N;
	std::vector<double> A;
	std::vector<double> B;
	bool transposed;
	// set to 0 to disable
	uint64_t block_input;
	uint64_t verbose;

	// TODO: why does this get called?
	multiplier() :
			N(0), transposed(false), block_input(0), verbose(0) {
	}

	multiplier(size_t N, std::vector<double> A,
			std::vector<double> B, bool transposed, uint64_t block_input,
			uint64_t verbose) :
	  N(N), A(A), B(B), //TODO: can I use std::move here? ask Hartmut
	  transposed(transposed), block_input(block_input), verbose(verbose)
  {
  }

	std::vector<double> calculate_submatrix(std::uint64_t x, std::uint64_t y,
			size_t block_result);

	HPX_DEFINE_COMPONENT_ACTION(multiplier, calculate_submatrix,
			calculate_submatrix_action);

};

}

HPX_REGISTER_ACTION_DECLARATION(
    multiply_components::multiplier::calculate_submatrix_action);
