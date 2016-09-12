#pragma once

#include <vector>
#include <cstddef>

struct matrix_multiply_work {
  size_t x;
  size_t y;
  size_t N;
  matrix_multiply_work(size_t x, size_t y, size_t N): x(x), y(y), N(N) {
  }
};

// uses round-robin distribution scheme, granularity of distribution is determined by the number of nodes
class matrix_multiply_static {
public:
  void insert_submatrix(std::vector<double> &C, std::vector<double> &submatrix, size_t N, matrix_multiply_work &w);
  
  std::vector<matrix_multiply_work> create_work_packages(size_t N, size_t num_localities);
  
  std::vector<double> matrix_multiply(size_t N, std::vector<double> &A, std::vector<double> &B);
};
