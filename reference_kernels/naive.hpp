/*
 * naive_matrix_m\ultiplication.cpp
 *
 *  Created on: Sep 5, 2016
 *      Author: pfandedd
 */

#pragma once

#include <cinttypes>

template <typename T>
std::vector<T> naive_matrix_multiply(std::size_t N, std::vector<T> &A,
                                     std::vector<T> &B) {
  std::vector<T> C(N * N);
#pragma omp parallel for
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      T result_component = 0.0;
      for (uint64_t k = 0; k < N; k++) {
        result_component += A.at(i * N + k) * B.at(k * N + j);
      }
      C.at(i * N + j) = result_component;
    }
  }
  return C;
}

// matrix multiply with matrix B assumed transposed
template <typename T>
std::vector<T> naive_matrix_multiply_transposed(std::size_t N,
                                                std::vector<T> &A,
                                                std::vector<T> &B) {
  std::vector<T> C(N * N);
#pragma omp parallel for
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      T result_component = 0.0;
      for (uint64_t k = 0; k < N; k++) {
        result_component += A.at(i * N + k) * B.at(j * N + k);
      }
      C.at(i * N + j) = result_component;
    }
  }
  return C;
}
