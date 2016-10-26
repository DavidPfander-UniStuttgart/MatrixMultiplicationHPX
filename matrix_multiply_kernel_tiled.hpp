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

namespace kernel_tiled {

  class matrix_multiply_kernel_tiled {
  private:
    std::size_t N_org;
    std::size_t X_size;
    std::size_t Y_size;
    std::size_t K_size;
    std::vector<double> A; 
    std::vector<double> B;

    uint64_t repetitions;
    uint64_t verbose;

    void verify_blocking_setup();
  public:
    matrix_multiply_kernel_tiled(size_t N, std::vector<double> &A_org,
				 std::vector<double> &B_org, bool transposed,
				 uint64_t repetitions, uint64_t verbose);
    
    std::vector<double> matrix_multiply(double &duration);
  };

}
