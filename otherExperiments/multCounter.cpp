#include <cstddef>
#include <cstdint>

void multCounter(double *A, double *B, double *C, const uint32_t n) {

  #pragma omp parallel for
  for (size_t i = 0; i < n; i += 1) {
    for (size_t j = 0; j < n; j += 1) {
      double result = 0.0;
      for (size_t k = 0; k < n; k += 1) {
	result += A[i * n + k] * B[j * n + k];
      }
      C[i * n + j] = result;
    }
  }
}

