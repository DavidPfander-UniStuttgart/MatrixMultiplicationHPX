#include <cstddef>
#include <cstdint>

#include <omp.h>

#define BLOCKSIZE 128

#define KCHUNK 128

void multGPU(double *A, double *B, double *C, const uint32_t n) {

  // omp_set_num_threads(4);

  #pragma omp parallel for
  for (size_t i = 0; i < n; i += BLOCKSIZE) {
    for (size_t j = 0; j < n; j += BLOCKSIZE) {

      double result[BLOCKSIZE * BLOCKSIZE];

      for (size_t ii = 0; ii < BLOCKSIZE; ii++) {
	for (size_t jj = 0; jj < BLOCKSIZE; jj++) {
	  result[ii * BLOCKSIZE + jj] = 0.0;
	}
      }

      for (size_t kBlock = 0; kBlock < n; kBlock += KCHUNK) {

	double AA[BLOCKSIZE * KCHUNK];
	for (size_t ii = 0; ii < BLOCKSIZE; ii++) {
	  for (size_t k = 0; k < KCHUNK; k++) {
	    AA[ii * KCHUNK + k] = A[(i +  ii) * n + kBlock + k];
	  }
	}

	double BB[BLOCKSIZE * KCHUNK];
	for (size_t jj = 0; jj < BLOCKSIZE; jj++) {
	  for (size_t k = 0; k < KCHUNK; k++) {
	    BB[jj * KCHUNK + k] = B[(j + jj) * n + kBlock + k];
	  }
	}

	/*
	  Inner block multiplication
	 */
	// for (size_t ii = 0; ii < BLOCKSIZE; ii += 4) {
	for (size_t ii = 0; ii < BLOCKSIZE; ii += 1) {
	  // for (size_t jj = 0; jj < BLOCKSIZE; jj += 8) {
	  for (size_t jj = 0; jj < BLOCKSIZE; jj += 1) {

	    double resultChunk1[4]; //only one register and single instruction for initialization

	    for (size_t kk = 0; kk < 4; kk++) {
	      resultChunk1[kk] = 0.0;
	    }

	    for (size_t k = 0; k < KCHUNK; k += 4) {
	      for (size_t kk = 0; kk < 4; kk++) { //one SIMD lane
		resultChunk1[kk] += AA[(ii + 0) * KCHUNK + k + kk] * BB[(jj + 0) * KCHUNK + k + kk];
	      }
	    }

	    for (size_t kk = 0; kk < 4; kk++) { //horizontal sum + single store instruction
	      result[(ii + 0) * BLOCKSIZE + jj + 0] += resultChunk1[kk];
	    }
	  }
	}
      }

      for (size_t ii = 0; ii < BLOCKSIZE; ii++) {
	for (size_t jj = 0; jj < BLOCKSIZE; jj++) {
	  C[(i + ii) * n + (j + jj)] = result[ii * BLOCKSIZE + jj];
	}
      }

    }
  }
}

