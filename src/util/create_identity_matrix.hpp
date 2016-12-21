#include <random>
#include <vector>

namespace util {

template <typename T> std::vector<T> create_identity_matrix(size_t N) {

  std::vector<T> m(N * N);
	std::fill(m.begin(), m.end(), 0.0);

	for (size_t i = 0; i < N; i++) {
		m[i * N + i] = 1.0;
	}

  return m;
}
}
