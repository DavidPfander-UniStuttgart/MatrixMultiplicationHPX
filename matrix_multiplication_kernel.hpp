/*
 *  matrix_multiplication_kernel.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: pfandedd
 */

#pragma once

namespace kernel {

// make async as well
template<typename T>
void extract_submatrix(std::vector<T> &C, std::vector<T> &C_small, size_t x,
		size_t y, size_t blockSize) {
	try {
		size_t blocksize_last = 2 * blockSize;
		for (uint64_t i = 0; i < blockSize; i++) {
			for (uint64_t j = 0; j < blockSize; j++) {
//				std::cout << "x: " << x << " y: " << y << " i: " << i << " j: "
//						<< j << " blockSize: " << blockSize << " N: " << N
//						<< std::endl;
				C.at((x + i) * blocksize_last + (y + j)) = C_small.at(
						i * blockSize + j);
			}
		}
	} catch (const std::out_of_range &oor) {
		std::cout << "in extract_submatrix: " << oor.what() << std::endl;
	}
}

template<typename T>
void matrix_multiply_kernel(std::vector<T> &A, std::vector<T> &B,
		std::vector<T> &C, size_t N, size_t x, size_t y, size_t blockSize) {
//    if (verbose >= 1) {
//        std::cout << "handling small matrix, x = " << x << ", y = " << y
//                << ", blocksize = " << blockSize << std::endl;
//    }


	try {
		for (uint64_t i = 0; i < blockSize; i++) {
			for (uint64_t j = 0; j < blockSize; j++) {
				for (uint64_t k = 0; k < N; k++) {
					C.at(i * blockSize + j) += A.at((x + i) * N + k)
							* B.at(k * N + (y + j));
				}
			}
		}
	} catch (const std::out_of_range &oor) {
		std::cout << "in matrix_multiply_kernel: \"" << oor.what() << "\"" << std::endl;
	}
}

}

