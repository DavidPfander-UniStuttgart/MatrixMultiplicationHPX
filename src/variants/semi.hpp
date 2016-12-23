#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace semi {

class semi {

private:
  size_t N;
  std::vector<double> &A;
  std::vector<double> &B;

  uint64_t block_result;
  uint64_t block_input;

public:
  semi(size_t N, std::vector<double> &A, std::vector<double> &B,
       uint64_t block_result, uint64_t block_input);

  std::vector<double> matrix_multiply();
};
}
