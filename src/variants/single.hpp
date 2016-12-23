#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace single {

class single {

private:
  size_t N;
  std::vector<double> &A;
  std::vector<double> &B;
  bool transposed;

  uint64_t block_result;
  uint64_t block_input;
  uint64_t repetitions;
  uint64_t verbose;

public:
  single(size_t N, std::vector<double> &A, std::vector<double> &B,
         bool transposed, uint64_t block_result, uint64_t block_input,
         uint64_t repetitions, uint64_t verbose);

  std::vector<double> matrix_multiply();
};
}
