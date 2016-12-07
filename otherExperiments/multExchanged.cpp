#include <cstddef>
#include <cstdint>
#include <x86intrin.h>

#include <omp.h>

#define BLOCKSIZEX 128
#define BLOCKSIZEY 128

#define KCHUNK 128

void multExchanged(double *A, double *B, double *C, const uint32_t n) {

  omp_set_num_threads(2);

  #pragma omp parallel for
  for (size_t i = 0; i < n; i += BLOCKSIZEX) {
    for (size_t j = 0; j < n; j += BLOCKSIZEY) {

      double result[BLOCKSIZEX * BLOCKSIZEY];

      for (size_t ii = 0; ii < BLOCKSIZEX; ii++) {
	for (size_t jj = 0; jj < BLOCKSIZEY; jj++) {
	  result[ii * BLOCKSIZEY + jj] = 0.0;
	}
      }

      for (size_t kBlock = 0; kBlock < n; kBlock += KCHUNK) {

	double AA[BLOCKSIZEX * KCHUNK];
	for (size_t ii = 0; ii < BLOCKSIZEX; ii++) {
	  for (size_t k = 0; k < KCHUNK; k++) {
	    AA[ii * KCHUNK + k] = A[(i +  ii) * n + kBlock + k];
	  }
	}

	double BB[BLOCKSIZEY * KCHUNK];
	for (size_t k = 0; k < KCHUNK; k++) {
	  for (size_t jj = 0; jj < BLOCKSIZEY; jj++) {
	    BB[jj * KCHUNK + k] = B[(kBlock + k) * n + (j + jj)];
	  }
	}

	for (size_t ii = 0; ii < BLOCKSIZEX; ii += 4) {
	  for (size_t k = 0; k < KCHUNK; k += 4) {

	    double Aregister1[4];
	    for (size_t kk = 0; kk < 4; kk++) { //one SIMD lane
	      Aregister1[kk] = AA[(ii + 0) * KCHUNK + k + kk];
	    }

	    double Aregister2[4];
	    for (size_t kk = 0; kk < 4; kk++) { //one SIMD lane
	      Aregister2[kk] = AA[(ii + 1) * KCHUNK + k + kk];
	    }

	    double Aregister3[4];
	    for (size_t kk = 0; kk < 4; kk++) { //one SIMD lane
	      Aregister3[kk] = AA[(ii + 2) * KCHUNK + k + kk];
	    }

	    double Aregister4[4];
	    for (size_t kk = 0; kk < 4; kk++) { //one SIMD lane
	      Aregister4[kk] = AA[(ii + 3) * KCHUNK + k + kk];
	    }
	    
	    size_t indexResultBase1 = (ii + 0) * BLOCKSIZEY;
	    size_t indexResultBase2 = (ii + 1) * BLOCKSIZEY;
	    size_t indexResultBase3 = (ii + 2) * BLOCKSIZEY;
	    size_t indexResultBase4 = (ii + 3) * BLOCKSIZEY;

	    for (size_t jj = 0; jj < BLOCKSIZEY; jj += 4) {
	    
	      for (size_t kk = 0; kk < 4; kk++) { //one SIMD lane
		result[indexResultBase1 + jj + 0] += Aregister1[kk] * BB[(jj + 0) * KCHUNK + (k + kk)];
		result[indexResultBase1 + jj + 1] += Aregister1[kk] * BB[(jj + 1) * KCHUNK + (k + kk)];
		result[indexResultBase1 + jj + 2] += Aregister1[kk] * BB[(jj + 2) * KCHUNK + (k + kk)];
		result[indexResultBase1 + jj + 3] += Aregister1[kk] * BB[(jj + 3) * KCHUNK + (k + kk)];

		result[indexResultBase2 + jj + 0] += Aregister2[kk] * BB[(jj + 0) * KCHUNK + (k + kk)];
		result[indexResultBase2 + jj + 1] += Aregister2[kk] * BB[(jj + 1) * KCHUNK + (k + kk)];
		result[indexResultBase2 + jj + 2] += Aregister2[kk] * BB[(jj + 2) * KCHUNK + (k + kk)];
		result[indexResultBase2 + jj + 3] += Aregister2[kk] * BB[(jj + 3) * KCHUNK + (k + kk)];

		result[indexResultBase3 + jj + 0] += Aregister3[kk] * BB[(jj + 0) * KCHUNK + (k + kk)];
		result[indexResultBase3 + jj + 1] += Aregister3[kk] * BB[(jj + 1) * KCHUNK + (k + kk)];
		result[indexResultBase3 + jj + 2] += Aregister3[kk] * BB[(jj + 2) * KCHUNK + (k + kk)];
		result[indexResultBase3 + jj + 3] += Aregister3[kk] * BB[(jj + 3) * KCHUNK + (k + kk)];

		result[indexResultBase4 + jj + 0] += Aregister4[kk] * BB[(jj + 0) * KCHUNK + (k + kk)];
		result[indexResultBase4 + jj + 1] += Aregister4[kk] * BB[(jj + 1) * KCHUNK + (k + kk)];
		result[indexResultBase4 + jj + 2] += Aregister4[kk] * BB[(jj + 2) * KCHUNK + (k + kk)];
		result[indexResultBase4 + jj + 3] += Aregister4[kk] * BB[(jj + 3) * KCHUNK + (k + kk)];
	      }
	    }
	    
	  }
	}
      }

      for (size_t ii = 0; ii < BLOCKSIZEX; ii++) {
	for (size_t jj = 0; jj < BLOCKSIZEY; jj++) {
	  C[(i + ii) * n + (j + jj)] = result[ii * BLOCKSIZEY + jj];
	}
      }

    }
  }

}

