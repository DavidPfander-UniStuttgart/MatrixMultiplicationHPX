#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kernel_test {

  class kernel_test {
  private:
    size_t N;
    std::vector<double> &A;
    std::vector<double> &B;

    uint64_t repetitions;
    uint64_t verbose;
  public:
    kernel_test(size_t N, std::vector<double> &A,
				std::vector<double> &B, bool transposed,
				uint64_t repetitions, uint64_t verbose);

    std::vector<double> matrix_multiply();
  };

}
