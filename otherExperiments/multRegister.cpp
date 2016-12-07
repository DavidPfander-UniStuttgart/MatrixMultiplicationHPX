#include <cstddef>
#include <cstdint>
#include <x86intrin.h>

#include <omp.h>

#define BLOCKSIZEX 256
#define BLOCKSIZEY 256

#define KCHUNK 64

void multRegister(double *A, double *B, double *C, const uint32_t n) {

  omp_set_num_threads(2);

  //iterate through the blocks in the result matrix
  #pragma omp parallel for
  for (size_t i = 0; i < n; i += BLOCKSIZEX) {
    for (size_t j = 0; j < n; j += BLOCKSIZEY) {

      double result[BLOCKSIZEX * BLOCKSIZEY];

      for (size_t ii = 0; ii < BLOCKSIZEX; ii++) {
      	for (size_t jj = 0; jj < BLOCKSIZEY; jj++) {
      	  result[ii * BLOCKSIZEY + jj] = 0.0;
      	}
      }

      //iterate through the "band" for the current result block
      for (size_t kBlock = 0; kBlock < n; kBlock += KCHUNK) {

	double AA[BLOCKSIZEX * KCHUNK];
	for (size_t ii = 0; ii < BLOCKSIZEX; ii++) {
	  for (size_t k = 0; k < KCHUNK; k++) {

	    AA[ii * KCHUNK + k] = A[(i +  ii) * n + (kBlock + k)];
	    // AA[k * BLOCKSIZEX + ii] = A[(kBlock + k) * n + (i +  ii)];
	  }
	}

	double BB[BLOCKSIZEY * KCHUNK];
	for (size_t k = 0; k < KCHUNK; k++) {
	  for (size_t jj = 0; jj < BLOCKSIZEY; jj++) {
	    BB[k * BLOCKSIZEY + jj] = B[(kBlock + k) * n + (j + jj)];
	  }
	}
	
	#define II_BLOCKSIZE 8
	#define JJ_BLOCKSIZE 2
	
	//calculate the current block
	for (size_t ii = 0; ii < BLOCKSIZEX; ii += II_BLOCKSIZE) {
	// for (size_t ii = 0; ii < BLOCKSIZEX; ii += 4) {
	  // for (size_t jj = 0; jj < BLOCKSIZEY; jj += 16) {
	  for (size_t jj = 0; jj < BLOCKSIZEY; jj += JJ_BLOCKSIZE * 4) {

	    __m256d resultReg[II_BLOCKSIZE * JJ_BLOCKSIZE];


	    for (size_t iii = 0; iii < II_BLOCKSIZE; iii++) {
	      for (size_t jjj = 0; jjj < JJ_BLOCKSIZE; jjj += 1) {
		resultReg[iii * JJ_BLOCKSIZE + jjj] = _mm256_load_pd(result + (ii + iii) * BLOCKSIZEY + jj + (jjj * 4));	      
	      }
	    }

	    for (size_t k = 0; k < KCHUNK; k += 1) {
	      // for (size_t k = 0; k < KCHUNK; k += 4) {
	      for (size_t iii = 0; iii < II_BLOCKSIZE; iii++) {	      
		__m256d Aregister1 = _mm256_set_pd(AA[(ii + iii) * KCHUNK + (k + 0)],
						   AA[(ii + iii) * KCHUNK + (k + 0)],
						   AA[(ii + iii) * KCHUNK + (k + 0)],
						   AA[(ii + iii) * KCHUNK + (k + 0)]);

		for (size_t jjj = 0; jjj < JJ_BLOCKSIZE; jjj += 1) {
		  __m256d Bregister1 = _mm256_load_pd(BB + (k + 0) * BLOCKSIZEY + jj + (jjj * 4));
#ifdef __FMA__
		  resultReg[iii * JJ_BLOCKSIZE + jjj] = _mm256_fmadd_pd(Aregister1, Bregister1, resultReg[iii * JJ_BLOCKSIZE + jjj]);
#else
		  __m256d temp = _mm256_mul_pd(Aregister1, Bregister1);
		  resultReg[jjj * JJ_BLOCKSIZE + iii] = _mm256_add_pd(temp, resultReg[jjj * JJ_BLOCKSIZE + iii]);		  
#endif
		}	      

		/*		Aregister1 = _mm256_set_pd(AA[(ii + iii) * KCHUNK + (k + 1)],
					   AA[(ii + iii) * KCHUNK + (k + 1)],
					   AA[(ii + iii) * KCHUNK + (k + 1)],
					   AA[(ii + iii) * KCHUNK + (k + 1)]);

		for (size_t jjj = 0; jjj < JJ_BLOCKSIZE; jjj += 1) {
		  __m256d Bregister1 = _mm256_load_pd(BB + (k + 1) * BLOCKSIZEY + jj + (jjj * 4));
#ifdef __FMA__
		  resultReg[iii * JJ_BLOCKSIZE + jjj] = _mm256_fmadd_pd(Aregister1, Bregister1, resultReg[iii * JJ_BLOCKSIZE + jjj]);
#else
		  __m256d temp = _mm256_mul_pd(Aregister1, Bregister1);
		  resultReg[jjj * JJ_BLOCKSIZE + iii] = _mm256_add_pd(temp, resultReg[jjj * JJ_BLOCKSIZE + iii]);		  
#endif
		}	      

		Aregister1 = _mm256_set_pd(AA[(ii + iii) * KCHUNK + (k + 2)],
					   AA[(ii + iii) * KCHUNK + (k + 2)],
					   AA[(ii + iii) * KCHUNK + (k + 2)],
					   AA[(ii + iii) * KCHUNK + (k + 2)]);

		for (size_t jjj = 0; jjj < JJ_BLOCKSIZE; jjj += 1) {
		  __m256d Bregister1 = _mm256_load_pd(BB + (k + 2) * BLOCKSIZEY + jj + (jjj * 4));
#ifdef __FMA__
		  resultReg[iii * JJ_BLOCKSIZE + jjj] = _mm256_fmadd_pd(Aregister1, Bregister1, resultReg[iii * JJ_BLOCKSIZE + jjj]);
#else
		  __m256d temp = _mm256_mul_pd(Aregister1, Bregister1);
		  resultReg[jjj * JJ_BLOCKSIZE + iii] = _mm256_add_pd(temp, resultReg[jjj * JJ_BLOCKSIZE + iii]);		  
#endif
		}	      

		Aregister1 = _mm256_set_pd(AA[(ii + iii) * KCHUNK + (k + 3)],
					   AA[(ii + iii) * KCHUNK + (k + 3)],
					   AA[(ii + iii) * KCHUNK + (k + 3)],
					   AA[(ii + iii) * KCHUNK + (k + 3)]);

		for (size_t jjj = 0; jjj < JJ_BLOCKSIZE; jjj += 1) {
		  __m256d Bregister1 = _mm256_load_pd(BB + (k + 3) * BLOCKSIZEY + jj + (jjj * 4));
#ifdef __FMA__
		  resultReg[iii * JJ_BLOCKSIZE + jjj] = _mm256_fmadd_pd(Aregister1, Bregister1, resultReg[iii * JJ_BLOCKSIZE + jjj]);
#else
		  __m256d temp = _mm256_mul_pd(Aregister1, Bregister1);
		  resultReg[jjj * JJ_BLOCKSIZE + iii] = _mm256_add_pd(temp, resultReg[jjj * JJ_BLOCKSIZE + iii]);		  
#endif
}*/	      
		}
	    }


// 	    for (size_t k = 0; k < KCHUNK; k += 1) {
	      
// 	      // __m256d Aregister1 = _mm256_set_pd(AA[(ii + 0) * KCHUNK + k + 0],
// 	      // 					 AA[(ii + 0) * KCHUNK + k + 0],
// 	      // 					 AA[(ii + 0) * KCHUNK + k + 0],
// 	      // 					 AA[(ii + 0) * KCHUNK + k + 0]);

// 	      for (size_t jjj = 0; jjj < 16; jjj += 4) {
// 		__m256d Bregister1 = _mm256_load_pd(BB + (k + 0) * BLOCKSIZEY + jj + jjj);
// 		for (size_t iii = 0; iii < 4; iii++) {	      
// 		  __m256d Aregister1 = _mm256_set_pd(AA[(ii + iii) * KCHUNK + k + 0],
// 						     AA[(ii + iii) * KCHUNK + k + 0],
// 						     AA[(ii + iii) * KCHUNK + k + 0],
// 						     AA[(ii + iii) * KCHUNK + k + 0]);


// #ifdef __FMA__
// 		  resultReg[jjj + iii] = _mm256_fmadd_pd(Aregister1, Bregister1, resultReg[jjj + iii]);
// #else
// 		  __m256d temp = _mm256_mul_pd(Aregister1, Bregister1);
// 		  resultReg[jjj + iii] = _mm256_add_pd(temp, resultReg[jjj + iii]);
		  
// #endif
// 		}
// 	      }
// 	    }

	    //update result temporary storage
	    for (size_t iii = 0; iii < II_BLOCKSIZE; iii++) {
	      // for (size_t jjj = 0; jjj < 16; jjj += 4) {
	      // for (size_t jjj = 0; jjj < 4; jjj += 1) {
	      for (size_t jjj = 0; jjj < JJ_BLOCKSIZE; jjj += 1) {
		// _mm256_store_pd(result + (ii + iii) * BLOCKSIZEY + jj + (jjj * 4), resultReg[iii * 4 + jjj]);
		_mm256_store_pd(result + (ii + iii) * BLOCKSIZEY + jj + (jjj * 4), resultReg[iii * JJ_BLOCKSIZE + jjj]);
	      }
	    }
	  }
	}
      }

      //write result of current block back to result matrix
      for (size_t ii = 0; ii < BLOCKSIZEX; ii++) {
      	for (size_t jj = 0; jj < BLOCKSIZEY; jj++) {
      	  C[(i + ii) * n + (j + jj)] = result[ii * BLOCKSIZEY + jj];
      	}
      }

    }
  }

}

