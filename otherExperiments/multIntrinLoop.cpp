#include <cstddef>
#include <cstdint>
#include <x86intrin.h>

#include <omp.h>
#include <iostream>

#define BLOCKSIZEX 128
#define BLOCKSIZEY 128

#define KCHUNK 128

void multIntrinLoop(double *A, double *B, double *C, const uint32_t n) {

  // omp_set_num_threads(2);

  #pragma omp parallel for
  for (size_t i = 0; i < n; i += BLOCKSIZEX) {
    for (size_t j = 0; j < n; j += BLOCKSIZEY) {

      alignas(32) double result[BLOCKSIZEX * BLOCKSIZEY];

      __m256d zero = _mm256_set1_pd(0.0);
      for (size_t ii = 0; ii < BLOCKSIZEX; ii++) {
	for (size_t jj = 0; jj < BLOCKSIZEY; jj += 4) {
      	// for (size_t jj = 0; jj < BLOCKSIZEY; jj++) {
      	  // result[ii * BLOCKSIZEY + jj] = 0.0;
	  _mm256_store_pd(result + ii * BLOCKSIZEY + jj, zero);
      	}
      }

      for (size_t kBlock = 0; kBlock < n; kBlock += KCHUNK) {

	alignas(32) double AA[BLOCKSIZEX * KCHUNK];
	for (size_t ii = 0; ii < BLOCKSIZEX; ii++) {
	  for (size_t k = 0; k < KCHUNK; k += 4) {
	  // for (size_t k = 0; k < KCHUNK; k++) {
	    // __m256d tmp = _mm256_load_pd(A + (i + ii) * n + (kBlock + k));
	    _mm256_store_pd(AA + ii * KCHUNK + k, _mm256_load_pd(A + (i + ii) * n + (kBlock + k)));
	    // AA[ii * KCHUNK + k] = A[(i + ii) * n + (kBlock + k)];
	    // AA[k * BLOCKSIZEX + ii] = A[(kBlock + k) * n + (i +  ii)];
	  }
	}

	alignas(32) double BB[BLOCKSIZEY * KCHUNK];
	for (size_t k = 0; k < KCHUNK; k++) {
	  // for (size_t jj = 0; jj < BLOCKSIZEY; jj++) {
	  for (size_t jj = 0; jj < BLOCKSIZEY; jj += 4) {
	    _mm256_store_pd(BB + k * BLOCKSIZEY + jj, _mm256_load_pd(B + (kBlock + k) * n + (j + jj)));
	    
	    // BB[jj * KCHUNK + k] = B[(j + jj) * n + (kBlock + k)];
	    // BB[k * BLOCKSIZEY + jj] = B[(kBlock + k) * n + (j + jj)];
	  }
	}

#define REG_BLOCK_II 4
#define REG_BLOCK_JJ 4
#define REG_BLOCK_JJ_STEP 4
#define REG_BLOCK_II_STRIDE (REG_BLOCK_II / REG_BLOCK_JJ_STEP)
#define REG_BLOCK_JJ_SIZE (REG_BLOCK_JJ * 4)

	for (size_t ii = 0; ii < BLOCKSIZEX; ii += REG_BLOCK_II) {
	  for (size_t jj = 0; jj < BLOCKSIZEY; jj += REG_BLOCK_JJ_SIZE) {	      

	    alignas(32) __m256d resultReg[REG_BLOCK_II * REG_BLOCK_JJ];
	    
	    for (size_t jjj = 0; jjj < REG_BLOCK_JJ_SIZE; jjj += REG_BLOCK_JJ_STEP) {
	      for (size_t iii = 0; iii < REG_BLOCK_II; iii++) {
		resultReg[jjj * REG_BLOCK_II_STRIDE + iii] = _mm256_load_pd(result + (ii + iii) * BLOCKSIZEY + jj + jjj);
	      }
	    }	    

	    for (size_t k = 0; k < KCHUNK; k += 1) {
	      
	      alignas(32) __m256d Aregister[REG_BLOCK_II];
	      for (size_t iii = 0; iii < REG_BLOCK_II; iii++) {
		Aregister[iii] = _mm256_set_pd(AA[(ii + iii) * KCHUNK + k + 0],
					       AA[(ii + iii) * KCHUNK + k + 0],
					       AA[(ii + iii) * KCHUNK + k + 0],
					       AA[(ii + iii) * KCHUNK + k + 0]);

		// Aregister[iii] = _mm256_set_pd(AA[(k + 0) * BLOCKSIZEX + (ii + iii)],
		// 			       AA[(k + 0) * BLOCKSIZEX + (ii + iii)],
		// 			       AA[(k + 0) * BLOCKSIZEX + (ii + iii)],
		// 			       AA[(k + 0) * BLOCKSIZEX + (ii + iii)]);

	      }

	      for (size_t jjj = 0; jjj < REG_BLOCK_JJ_SIZE; jjj += REG_BLOCK_JJ_STEP) {

		// __m256d Bregister1 = _mm256_load_pd(BB + (jj + jjj) * KCHUNK + (k + 0));
		// alignas(32) __m256d Bregister1 = _mm256_loadu_pd(BB + (k + 0) * BLOCKSIZEY + (jj + jjj));
		alignas(32) __m256d Bregister1 = _mm256_load_pd(BB + (k + 0) * BLOCKSIZEY + (jj + jjj));
		for (size_t iii = 0; iii < REG_BLOCK_II; iii++) {
#ifdef __FMA__
		  resultReg[jjj * REG_BLOCK_II_STRIDE + iii] = _mm256_fmadd_pd(Aregister[iii], Bregister1, resultReg[jjj * REG_BLOCK_II_STRIDE + iii]);
#else
		  alignas(32) __m256d temp = _mm256_mul_pd(Aregister[iii], Bregister1);
		  resultReg[jjj * REG_BLOCK_II_STRIDE + iii] = _mm256_add_pd(temp, resultReg[jjj * REG_BLOCK_II_STRIDE + iii]);
		  
#endif
		}
	      }

	    }

	    for (size_t iii = 0; iii < REG_BLOCK_II; iii++) {
	      for (size_t jjj = 0; jjj < REG_BLOCK_JJ_SIZE; jjj += REG_BLOCK_JJ_STEP) {
		_mm256_store_pd(result + (ii + iii) * BLOCKSIZEY + jj + jjj, resultReg[jjj * REG_BLOCK_II_STRIDE + iii]);
	      }
	    }
	  }
	}
      }

      for (size_t ii = 0; ii < BLOCKSIZEX; ii++) {
	// for (size_t jj = 0; jj < BLOCKSIZEY; jj++) {
	for (size_t jj = 0; jj < BLOCKSIZEY; jj += 4) {
	  _mm256_stream_pd(C + (i + ii) * n + (j + jj), _mm256_load_pd(result + ii * BLOCKSIZEY + jj));
	  // C[(i + ii) * n + (j + jj)] = result[ii * BLOCKSIZEY + jj];
	}
      }

    }
  }

}

