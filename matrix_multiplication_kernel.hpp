/*
 *  matrix_multiplication_kernel.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: pfandedd
 */

namespace kernel {

// make async as well
template<typename T>
void extract_submatrix(std::vector<T> &C, std::vector<T> &C_small, size_t N,
        size_t x, size_t y, size_t blockSize) {
    for (uint64_t i = 0; i < blockSize; i++) {
        for (uint64_t j = 0; j < blockSize; j++) {
            C[(x + i) * N + (y + j)] = C_small[i * blockSize + j];
        }
    }
}

template<typename T>
std::vector<T> matrix_multiply_kernel(std::vector<T> &A, std::vector<T> &B,
        std::vector<T> &C, size_t N, size_t x, size_t y, size_t blockSize) {
//    if (verbose >= 1) {
//        std::cout << "handling small matrix, x = " << x << ", y = " << y
//                << ", blocksize = " << blockSize << std::endl;
//    }

    for (uint64_t i = 0; i < blockSize; i++) {
        for (uint64_t j = 0; j < blockSize; j++) {
            for (uint64_t k = 0; k < N; k++) {
                C[i * blockSize + j] += A[(x + i) * N + k] * B[k * N + (y + j)];
            }
        }
    }
    return C;
}

}



