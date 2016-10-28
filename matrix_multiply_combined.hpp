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
    std::size_t N_org;
    std::size_t X_size;
    std::size_t Y_size;
    std::size_t K_size;

    std::vector<double> A;
    std::vector<double> B;
    bool transposed;

    uint64_t block_result;
    uint64_t block_input;
    uint64_t repetitions;
    uint64_t verbose;

    void verify_blocking_setup();
  public:
    matrix_multiply_combined(size_t N, std::vector<double> &A, std::vector<double> &B,
			     bool transposed, uint64_t block_result, uint64_t block_input,
			     uint64_t repetitions, uint64_t verbose);

    std::vector<double> matrix_multiply(double &duration);
  };
}
