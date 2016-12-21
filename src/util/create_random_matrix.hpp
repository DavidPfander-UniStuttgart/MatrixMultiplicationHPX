#include <random>
#include <vector>
#include <functional>
#include <algorithm>

namespace util {

template <typename T> std::vector<T> create_random_matrix(size_t N) {

  std::vector<T> m(N * N);

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  auto myRand = std::bind(distribution, generator);

  std::generate(m.begin(), m.end(), myRand);
  return m;
}
}
