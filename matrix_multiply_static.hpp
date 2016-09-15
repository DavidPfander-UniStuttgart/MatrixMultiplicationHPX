#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

#include "matrix_multiply_work.hpp"

// uses round-robin distribution scheme, granularity of distribution is determined by the number of nodes
class matrix_multiply_static {
private:
	size_t N;
	std::vector<double> &A;
	std::vector<double> &B;
	bool transposed;
	uint64_t block_input;
	std::vector<double> C;
	size_t small_block_size;
	uint64_t verbose;
public:
	matrix_multiply_static(size_t N, std::vector<double> &A,
			std::vector<double> &B, bool transposed, uint64_t block_input, size_t small_block_size,
			uint64_t verbose) :
			N(N), A(A), B(B), transposed(transposed), block_input(block_input), C(N * N), small_block_size(
					small_block_size), verbose(verbose) {
	}

	void insert_submatrix(const std::vector<double> &submatrix,
			const matrix_multiply_work &w);

	std::vector<matrix_multiply_work> create_work_packages(
			size_t num_localities);

	std::vector<double> matrix_multiply();
};
