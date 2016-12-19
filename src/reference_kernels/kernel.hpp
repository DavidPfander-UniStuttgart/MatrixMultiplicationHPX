#pragma once

//#include "aligned_allocator.hpp"
#include <boost/align/aligned_allocator.hpp>

namespace kernel {

template <typename T>
void kernel(std::vector<T> &A, std::vector<T> &B, std::vector<T> &C, size_t N,
            size_t x, size_t y, size_t blockSize) {
  try {
    for (uint64_t i = 0; i < blockSize; i++) {
      for (uint64_t j = 0; j < blockSize; j++) {
        T result_component = 0.0;
        for (uint64_t k = 0; k < N; k++) {
          result_component += A.at((x + i) * N + k) * B.at(k * N + (y + j));
        }
        C.at(i * blockSize + j) = result_component;
      }
    }
  } catch (const std::out_of_range &oor) {
    std::cout << "in kernel: \"" << oor.what() << "\"" << std::endl;
  }
}

// matrix multiply with matrix B assumed transposed
// OPT: assume matrix B transposed
// OPT: use accumulator for result in innermost loop
template <typename T>
void kernel_transposed(std::vector<T> &A, std::vector<T> &B, std::vector<T> &C,
                       size_t N, size_t x, size_t y, size_t blockSize) {
  for (uint64_t i = 0; i < blockSize; i++) {
    for (uint64_t j = 0; j < blockSize; j++) {
      T result_component = 0.0;
      for (uint64_t k = 0; k < N; k++) {
        result_component += A[(x + i) * N + k] * B[(y + j) * N + k];
      }
      C[i * blockSize + j] = result_component;
    }
  }
}

// matrix multiply with matrix B assumed transposed
// OPT: assume matrix B transposed
// OPT: use accumulator for result in innermost loop
// OPT: blocked loading of the input matrices -> data will be in cache nearly
// all of the time
// OPT: use unsafe accesses
// opt: precache data in small arrays to avoid large power of 2 conflict misses
// for large input arrays
template <typename T>
void kernel_transposed_blocked(std::vector<T> &A, std::vector<T> &B,
                               std::vector<T> &C, const size_t N,
                               const size_t x, const size_t y,
                               const size_t block_result,
                               const size_t block_input) {

  std::vector<T, boost::alignment::aligned_allocator<T, 32>> A_small(
      block_input * block_result);
  std::vector<T, boost::alignment::aligned_allocator<T, 32>> B_small(
      block_input * block_result);

  //    T *A_small = static_cast<T
  //    *>(__builtin_assume_aligned(A_small_aligned.data(), 32));
  //    T *B_small = static_cast<T
  //    *>(__builtin_assume_aligned(B_small_aligned.data(), 32));

  //    std::vector<T> C_small(block_result * block_result);

  // can skip two outer loops due to the implicit blocking because of the
  // recursive parallelization
  for (size_t k_block = 0; k_block < N; k_block += block_input) {

    //        size_t k_max = std::min(k_block + block_input, N);

    for (size_t i = 0; i < block_result; i++) {
      for (size_t k = 0; k < block_input; k++) {
        A_small[i * block_input + k] = A[(x + i) * N + k_block + k];
      }
    }

    for (size_t j = 0; j < block_result; j++) {
      for (size_t k = 0; k < block_input; k++) {
        B_small[j * block_input + k] = B[(y + j) * N + k_block + k];
      }
    }

    //        for (size_t i = 0; i < block_result; i++) {
    //            for (size_t j = 0; j < block_result; j++) {
    for (size_t i = 0; i < block_result; i += 4) {
      for (size_t j = 0; j < block_result; j += 2) {
        //                T result_component = 0.0;
        //                for (size_t k = 0; k < block_input; k++) {
        //                    result_component += A_small[i * block_input + k]
        //                            * B_small[j * block_input + k];
        //                }
        //                // assumes matrix was zero-initialized
        //                C[i * block_result + j] += result_component;
        T result_component_0_0 = 0.0;
        T result_component_0_1 = 0.0;
        T result_component_1_0 = 0.0;
        T result_component_1_1 = 0.0;
        T result_component_2_0 = 0.0;
        T result_component_2_1 = 0.0;
        T result_component_3_0 = 0.0;
        T result_component_3_1 = 0.0;

        for (size_t k = 0; k < block_input; k++) {
          result_component_0_0 += A_small[(i + 0) * block_input + k] *
                                  B_small[(j + 0) * block_input + k];
          result_component_0_1 += A_small[(i + 0) * block_input + k] *
                                  B_small[(j + 1) * block_input + k];
          result_component_1_0 += A_small[(i + 1) * block_input + k] *
                                  B_small[(j + 0) * block_input + k];
          result_component_1_1 += A_small[(i + 1) * block_input + k] *
                                  B_small[(j + 1) * block_input + k];

          result_component_2_0 += A_small[(i + 2) * block_input + k] *
                                  B_small[(j + 0) * block_input + k];
          result_component_2_1 += A_small[(i + 2) * block_input + k] *
                                  B_small[(j + 1) * block_input + k];
          result_component_3_0 += A_small[(i + 3) * block_input + k] *
                                  B_small[(j + 0) * block_input + k];
          result_component_3_1 += A_small[(i + 3) * block_input + k] *
                                  B_small[(j + 1) * block_input + k];
        }
        // assumes matrix was zero-initialized
        C[(i + 0) * block_result + (j + 0)] += result_component_0_0;
        C[(i + 0) * block_result + (j + 1)] += result_component_0_1;
        C[(i + 1) * block_result + (j + 0)] += result_component_1_0;
        C[(i + 1) * block_result + (j + 1)] += result_component_1_1;

        C[(i + 2) * block_result + (j + 0)] += result_component_2_0;
        C[(i + 2) * block_result + (j + 1)] += result_component_2_1;
        C[(i + 3) * block_result + (j + 0)] += result_component_3_0;
        C[(i + 3) * block_result + (j + 1)] += result_component_3_1;
      }
    }
  }

  //    for (size_t i = 0; i < block_result; i++) {
  //        for (size_t j = 0; j < block_result; j++) {
  //            C[i * block_result+ j] = C_small[i * block_result + j];
  //        }
  //    }
}
}
