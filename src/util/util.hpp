#pragma once

#include <hpx/include/iostreams.hpp>
#include <iostream>

template <typename T, typename Alloc>
void print_matrix_host(size_t N, const std::vector<T, Alloc> &m) {
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

template <typename T, typename Alloc>
void print_matrix_host(size_t rows, size_t cols,
                       const std::vector<T, Alloc> &m) {
  for (std::uint64_t r = 0; r < rows; r++) {
    for (std::uint64_t c = 0; c < cols; c++) {
      if (c > 0) {
        std::cout << ", ";
      }
      std::cout << m.at(r * cols + c);
    }
    std::cout << std::endl;
  }
}

template <typename T, typename Alloc>
void print_matrix_transposed_host(size_t N, const std::vector<T, Alloc> &m) {
  for (std::uint64_t i = 0; i < N; i++) {
    for (std::uint64_t j = 0; j < N; j++) {
      if (j > 0) {
        std::cout << ", ";
      }
      std::cout << m.at(j * N + i);
    }
    std::cout << std::endl;
  }
}

template <typename T, typename Alloc>
void print_matrix_transposed_host(size_t N, size_t M,
                                  const std::vector<T, Alloc> &m) {
  for (std::uint64_t i = 0; i < N; i++) {
    for (std::uint64_t j = 0; j < M; j++) {
      if (j > 0) {
        std::cout << ", ";
      }
      std::cout << m.at(j * N + i);
    }
    std::cout << std::endl;
  }
}

template <typename T, typename Alloc>
void print_matrix(size_t N, std::vector<T, Alloc> &m) {
  for (std::uint64_t i = 0; i < N; i++) {
    for (std::uint64_t j = 0; j < N; j++) {
      if (j > 0) {
        hpx::cout << ", ";
      }
      hpx::cout << m.at(i * N + j);
    }
    hpx::cout << std::endl << hpx::flush;
  }
}

template <typename T, typename Alloc>
void print_matrix(size_t N, size_t M, std::vector<T, Alloc> &m) {
  for (std::uint64_t i = 0; i < N; i++) {
    for (std::uint64_t j = 0; j < M; j++) {
      if (j > 0) {
        hpx::cout << ", ";
      }
      hpx::cout << m.at(i * N + j);
    }
    hpx::cout << std::endl << hpx::flush;
  }
}

template <typename T, typename Alloc>
void print_matrix_transposed(size_t N, std::vector<T, Alloc> &m) {
  for (std::uint64_t i = 0; i < N; i++) {
    for (std::uint64_t j = 0; j < N; j++) {
      if (j > 0) {
        hpx::cout << ", ";
      }
      hpx::cout << m.at(j * N + i);
    }
    hpx::cout << std::endl << hpx::flush;
  }
}

template <typename T, typename Alloc>
void print_matrix_transposed(size_t N, size_t M, std::vector<T, Alloc> &m) {
  for (std::uint64_t i = 0; i < N; i++) {
    for (std::uint64_t j = 0; j < M; j++) {
      if (j > 0) {
        hpx::cout << ", ";
      }
      hpx::cout << m.at(j * N + i);
    }
    hpx::cout << std::endl << hpx::flush;
  }
}
