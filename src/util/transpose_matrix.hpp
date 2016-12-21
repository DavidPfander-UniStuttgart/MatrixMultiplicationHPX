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
}
