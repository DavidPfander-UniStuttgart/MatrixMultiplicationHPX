/*
 * matrix_multiply_algorithm.hpp
 *
 *  Created on: Oct 10, 2016
 *      Author: pfandedd
 */

#pragma once

#include "hpx/parallel/algorithms/for_each.hpp"
#include "hpx/parallel/algorithms/for_loop.hpp"
#include "hpx/parallel/execution_policy.hpp"
#include <boost/iterator/iterator_facade.hpp>
#include "hpx/util/iterator_facade.hpp"

namespace kernel_test {

class matrix_multiply_kernel_test {

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
  matrix_multiply_kernel_test(size_t N, std::vector<double> &A,
      std::vector<double> &B, bool transposed, uint64_t block_result,
      uint64_t block_input, uint64_t repetitions, uint64_t verbose) :
      N(N), A(A), B(B), transposed(transposed), block_result(block_result), block_input(
          block_input), repetitions(repetitions), verbose(verbose) {

  }

  std::vector<double> matrix_multiply() {
    std::vector<double> C(N * N);
    std::fill(C.begin(), C.end(), 0.0);

    for (size_t rep = 0; rep < repetitions; rep++) {
      for (size_t x = 0; x < N; x += 4) {
        for (size_t y = 0; y < N; y += 2) {
//          for (size_t k = 0; k < N; k += block_input) {
          double result_component_0_0 = 0.0;
          double result_component_0_1 = 0.0;
          double result_component_1_0 = 0.0;
          double result_component_1_1 = 0.0;

          double result_component_2_0 = 0.0;
          double result_component_2_1 = 0.0;
          double result_component_3_0 = 0.0;
          double result_component_3_1 = 0.0;

//          double result_component_0_2 = 0.0;
//          double result_component_0_3 = 0.0;
//          double result_component_1_2 = 0.0;
//          double result_component_1_3 = 0.0;
//          double result_component_2_2 = 0.0;
//          double result_component_2_3 = 0.0;
//          double result_component_3_2 = 0.0;
//          double result_component_3_3 = 0.0;

          for (size_t k_inner = 0; k_inner < N; k_inner++) {

            result_component_0_0 += A[(x + 0) * N + k_inner]
                * B[(y + 0) * N + k_inner];
            result_component_0_1 += A[(x + 0) * N + k_inner]
                * B[(y + 1) * N + k_inner];
            result_component_1_0 += A[(x + 1) * N + k_inner]
                * B[(y + 0) * N + k_inner];
            result_component_1_1 += A[(x + 1) * N + k_inner]
                * B[(y + 1) * N + k_inner];

            result_component_2_0 += A[(x + 2) * N + k_inner]
                * B[(y + 0) * N + k_inner];
            result_component_2_1 += A[(x + 2) * N + k_inner]
                * B[(y + 1) * N + k_inner];
            result_component_3_0 += A[(x + 3) * N + k_inner]
                * B[(y + 0) * N + k_inner];
            result_component_3_1 += A[(x + 3) * N + k_inner]
                * B[(y + 1) * N + k_inner];

//            result_component_0_2 += A[(x + 0) * N + k_inner]
//                * B[(y + 2) * N + k_inner];
//            result_component_0_3 += A[(x + 0) * N + k_inner]
//                * B[(y + 3) * N + k_inner];
//            result_component_1_2 += A[(x + 1) * N + k_inner]
//                * B[(y + 2) * N + k_inner];
//            result_component_1_3 += A[(x + 1) * N + k_inner]
//                * B[(y + 3) * N + k_inner];
//
//            result_component_2_2 += A[(x + 2) * N + k_inner]
//                * B[(y + 2) * N + k_inner];
//            result_component_2_3 += A[(x + 2) * N + k_inner]
//                * B[(y + 3) * N + k_inner];
//            result_component_3_2 += A[(x + 3) * N + k_inner]
//                * B[(y + 2) * N + k_inner];
//            result_component_3_3 += A[(x + 3) * N + k_inner]
//                * B[(y + 3) * N + k_inner];
          }
          // assumes matrix was zero-initialized
          C[(x + 0) * N + (y + 0)] += result_component_0_0;
          C[(x + 0) * N + (y + 1)] += result_component_0_1;
          C[(x + 1) * N + (y + 0)] += result_component_1_0;
          C[(x + 1) * N + (y + 1)] += result_component_1_1;

          C[(x + 2) * N + (y + 0)] += result_component_2_0;
          C[(x + 2) * N + (y + 1)] += result_component_2_1;
          C[(x + 3) * N + (y + 0)] += result_component_3_0;
          C[(x + 3) * N + (y + 1)] += result_component_3_1;

//          C[(x + 0) * N + (y + 2)] += result_component_0_2;
//          C[(x + 0) * N + (y + 3)] += result_component_0_3;
//          C[(x + 1) * N + (y + 2)] += result_component_1_2;
//          C[(x + 1) * N + (y + 3)] += result_component_1_3;
//
//          C[(x + 2) * N + (y + 2)] += result_component_2_2;
//          C[(x + 2) * N + (y + 3)] += result_component_2_3;
//          C[(x + 3) * N + (y + 2)] += result_component_3_2;
//          C[(x + 3) * N + (y + 3)] += result_component_3_3;
//          }
        }
      }
    }

    return C;
  }
};

}
