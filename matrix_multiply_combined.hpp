/*
 * matrix_multiply_algorithm.hpp
 *
 *  Created on: Oct 10, 2016
 *      Author: pfandedd
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace combined {

  class matrix_multiply_combined {

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
    matrix_multiply_combined(size_t N, std::vector<double> &A, std::vector<double> &B,
			     bool transposed, uint64_t block_result, uint64_t block_input,
			     uint64_t repetitions, uint64_t verbose);

    std::vector<double> matrix_multiply();
  };
}
