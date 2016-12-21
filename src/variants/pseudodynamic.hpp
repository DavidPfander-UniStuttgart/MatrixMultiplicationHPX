#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pseudodynamic {

class pseudodynamic {

private:
  size_t N;
  std::vector<double> &A;
  std::vector<double> &B;
  bool transposed;

  uint64_t block_result;
  uint64_t block_input;

  std::uint64_t min_work_size;
  std::uint64_t max_work_difference; // TODO: shouldn't this be a double?
  double max_relative_work_difference;

  uint64_t repetitions;
  uint64_t verbose;

public:
  pseudodynamic(size_t N, std::vector<double> &A, std::vector<double> &B,
                bool transposed, uint64_t block_result, uint64_t block_input,
                size_t min_work_size, size_t max_work_difference,
                double max_relative_work_difference, uint64_t repetitions,
                uint64_t verbose);

  std::vector<double> matrix_multiply();
};
}
