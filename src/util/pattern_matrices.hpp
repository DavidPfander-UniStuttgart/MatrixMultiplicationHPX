#include <random>
#include <vector>

namespace util {

template <typename T> std::vector<T> create_enumerating_matrix(size_t N) {

  std::vector<T> m(N * N);

  for (size_t i = 0; i < N * N; i++) {
    m[i] = static_cast<double>(i);
  }

  return m;
}
}
