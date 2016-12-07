#include <cstddef>
#include <cstdint>

void multNaive(double *A, double *B, double *C, const uint32_t n) {

  #pragma omp parallel for
  for (size_t i = 0; i < n; i += 1) {
    for (size_t j = 0; j < n; j += 1) {
      for (size_t k = 0; k < n; k += 1) {
	C[i * n + j] += A[i * n + k] * B[k * n + j]; 
      }
    }
  }
}
