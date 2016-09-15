#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

#include "matrix_multiply_work.hpp"

// uses round-robin distribution scheme, granularity of distribution is determined by the number of nodes
class matrix_multiply_static_improved {
private:
	size_t N;
	std::vector<double> &A;
	std::vector<double> &B;
	std::vector<double> C;
	size_t small_block_size;

	uint64_t min_work_size;
	uint64_t max_work_difference;
	double max_relative_work_difference;

	uint64_t verbose;
public:
	matrix_multiply_static_improved(size_t N, std::vector<double> &A,
			std::vector<double> &B, size_t small_block_size,
			uint64_t min_work_size, uint64_t max_work_difference,
			double max_relative_work_difference, uint64_t verbose) :
			N(N), A(A), B(B), C(N * N), small_block_size(small_block_size), min_work_size(
					min_work_size), max_work_difference(max_work_difference), max_relative_work_difference(
					max_relative_work_difference), verbose(verbose) {
	}

	void print_schedule(
			std::vector<std::vector<matrix_multiply_work>> &all_work);

	void insert_submatrix(const std::vector<double> &submatrix,
			const matrix_multiply_work &w);

	bool fulfills_constraints(std::vector<uint64_t> &total_work);

	std::vector<std::vector<matrix_multiply_work>> create_schedule(
			size_t num_localities);

	std::vector<double> matrix_multiply();
};
