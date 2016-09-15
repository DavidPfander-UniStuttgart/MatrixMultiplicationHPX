/*
 *  matrix_multiplication_kernel.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: pfandedd
 */

#pragma once

namespace kernel {

template<typename T>
void matrix_multiply_kernel(std::vector<T> &A, std::vector<T> &B,
		std::vector<T> &C, size_t N, size_t x, size_t y, size_t blockSize) {
	try {
		for (uint64_t i = 0; i < blockSize; i++) {
			for (uint64_t j = 0; j < blockSize; j++) {
				T result_component = 0.0;
				for (uint64_t k = 0; k < N; k++) {
					result_component += A.at((x + i) * N + k)
							* B.at(k * N + (y + j));
				}
				C.at(i * blockSize + j) = result_component;
			}
		}
	} catch (const std::out_of_range &oor) {
		std::cout << "in matrix_multiply_kernel: \"" << oor.what() << "\""
				<< std::endl;
	}
}

// matrix multiply with matrix B assumed transposed
// OPT: assume matrix B transposed
// OPT: use accumulator for result in innermost loop
template<typename T>
void matrix_multiply_kernel_transposed(std::vector<T> &A, std::vector<T> &B,
		std::vector<T> &C, size_t N, size_t x, size_t y, size_t blockSize) {
	try {
		for (uint64_t i = 0; i < blockSize; i++) {
			for (uint64_t j = 0; j < blockSize; j++) {
				T result_component = 0.0;
				for (uint64_t k = 0; k < N; k++) {
					result_component += A[(x + i) * N + k]
							* B[(y + j) * N + k];
				}
				C[i * blockSize + j] = result_component;
			}
		}
	} catch (const std::out_of_range &oor) {
		std::cout << "in matrix_multiply_kernel: \"" << oor.what() << "\""
				<< std::endl;
	}
}

// matrix multiply with matrix B assumed transposed
// OPT: assume matrix B transposed
// OPT: use accumulator for result in innermost loop
// OPT: blocked loading of the input matrices -> data will be in cache nearly all of the time
// OPT: use unsafe accesses
template<typename T>
void matrix_multiply_kernel_transposed_blocked(std::vector<T> &A,
		std::vector<T> &B, std::vector<T> &C, const size_t N, const size_t x, const size_t y,
		const size_t block_result, const size_t block_input) {
	// can skip two outer loops due to the implicit blocking because of the recursive parallelization
	for (size_t k_block = 0; k_block < N; k_block += block_input) {
		for (size_t i = 0; i < block_result; i++) {
			for (size_t j = 0; j < block_result; j++) {
				T result_component = 0.0;
				for (size_t k = k_block; k < k_block + block_input; k++) {
					result_component += A[(x + i) * N + k] * B[(y + j) * N + k];
				}
				// assumes matrix was zero-initialized
				C[i * block_result + j] += result_component;
			}
		}
	}
}

}

