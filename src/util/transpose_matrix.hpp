#include <random>
#include <vector>

namespace util {

template <typename T>
std::vector<T> transpose_matrix(size_t N, std::vector<T> org) {

  std::vector<T> m(N * N);

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      m[i * N + j] = org[j * N + i];
    }
  }

  return m;
}

template <typename T, typename U>
std::vector<T, U> transpose_matrix(size_t stride_org, size_t stride_result,
                                   std::vector<T, U> org) {

  std::vector<T, U> result(stride_org * stride_result);

  for (size_t i = 0; i < stride_org; i++) {
    for (size_t j = 0; j < stride_result; j++) {
      result[i * stride_result + j] = org[j * stride_org + i];
    }
  }

  return result;
}
}
