/*
 * naive_matrix_m\ultiplication.cpp
 *
 *  Created on: Sep 5, 2016
 *      Author: pfandedd
 */

#pragma once

#include <cinttypes>

template<typename T>
std::vector<T> naiveMatrixMultiply(std::size_t N, std::vector<T> &A,
        std::vector<T> &B) {
    std::vector<T> C(N * N);
    for (uint64_t i = 0; i < N; i++) {
        for (uint64_t j = 0; j < N; j++) {
            for (uint64_t k = 0; k < N; k++) {
                C.at(i * N + j) += A.at(i * N + k) * B.at(k * N + j);
            }
        }
    }
    return C;
}

template<typename T>
void print_matrix(size_t N, std::vector<T> m) {
    for (std::uint64_t i = 0; i < N; i++) {
        for (std::uint64_t j = 0; j < N; j++) {
            if (j > 0) {
                std::cout << ", ";
            }
            std::cout << m.at(i * N + j);
        }
        std::cout << std::endl;
    }
}
