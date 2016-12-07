#include <cstddef>
#include <cstdint>

#include <omp.h>
#include <x86intrin.h>

#define BLOCKSIZE 256

#define KCHUNK 256

void multGPUASM(double *A, double *B, double *C, const uint32_t n) {

  typedef union avxValue {
    __m256d in;
    double out[4];
  } avxValue;

  // omp_set_num_threads(2);

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

	for (size_t ii = 0; ii < BLOCKSIZE; ii += 4) {
	  // for (size_t jj = 0; jj < BLOCKSIZE; jj++) {
	  for (size_t jj = 0; jj < BLOCKSIZE; jj += 4) {

	    __m256d resultChunk11 = _mm256_set1_pd(0.0);
	    __m256d resultChunk12 = _mm256_set1_pd(0.0);
	    __m256d resultChunk13 = _mm256_set1_pd(0.0);
	    __m256d resultChunk14 = _mm256_set1_pd(0.0);

	    __m256d resultChunk21 = _mm256_set1_pd(0.0);
	    __m256d resultChunk22 = _mm256_set1_pd(0.0);
	    __m256d resultChunk23 = _mm256_set1_pd(0.0);
	    __m256d resultChunk24 = _mm256_set1_pd(0.0);

	    __m256d resultChunk31 = _mm256_set1_pd(0.0);
	    __m256d resultChunk32 = _mm256_set1_pd(0.0);
	    __m256d resultChunk33 = _mm256_set1_pd(0.0);
	    __m256d resultChunk34 = _mm256_set1_pd(0.0);

	    __m256d resultChunk41 = _mm256_set1_pd(0.0);
	    __m256d resultChunk42 = _mm256_set1_pd(0.0);
	    __m256d resultChunk43 = _mm256_set1_pd(0.0);
	    __m256d resultChunk44 = _mm256_set1_pd(0.0);

	    // __m256d resultChunk21 = _mm256_set1_pd(0.0);
	    // __m256d resultChunk22 = _mm256_set1_pd(0.0);

	    // AA + BB -> requires 8 registers
	    // resultChunk -> requires 4 registers

	    for (size_t k = 0; k < KCHUNK; k += 4) {

	      // for (size_t kk = 0; kk < 4; kk++) { //one SIMD lane
	      // 	resultChunk1[kk] += AA[(ii + 0) * KCHUNK + k + kk] * BB[(jj + 0) * KCHUNK + k + kk];
	      // }

	      __m256d AAtemp1 = _mm256_load_pd(AA + (ii + 0) * KCHUNK + k);
	      __m256d AAtemp2 = _mm256_load_pd(AA + (ii + 1) * KCHUNK + k);
	      __m256d AAtemp3 = _mm256_load_pd(AA + (ii + 2) * KCHUNK + k);
	      __m256d AAtemp4 = _mm256_load_pd(AA + (ii + 3) * KCHUNK + k);

	      __m256d BBtemp1 = _mm256_load_pd(BB + (jj + 0) * KCHUNK + k);
	      __m256d BBtemp2 = _mm256_load_pd(BB + (jj + 1) * KCHUNK + k);
	      __m256d BBtemp3 = _mm256_load_pd(BB + (jj + 2) * KCHUNK + k);
	      __m256d BBtemp4 = _mm256_load_pd(BB + (jj + 3) * KCHUNK + k);

	      #ifdef __FMA__
	      resultChunk11 = _mm256_fmadd_pd(AAtemp1, BBtemp1, resultChunk11);
	      resultChunk12 = _mm256_fmadd_pd(AAtemp1, BBtemp2, resultChunk12);
	      resultChunk13 = _mm256_fmadd_pd(AAtemp1, BBtemp3, resultChunk13);
	      resultChunk14 = _mm256_fmadd_pd(AAtemp1, BBtemp4, resultChunk14);

	      resultChunk21 = _mm256_fmadd_pd(AAtemp2, BBtemp1, resultChunk21);
	      resultChunk22 = _mm256_fmadd_pd(AAtemp2, BBtemp2, resultChunk22);
	      resultChunk23 = _mm256_fmadd_pd(AAtemp2, BBtemp3, resultChunk23);
	      resultChunk24 = _mm256_fmadd_pd(AAtemp2, BBtemp4, resultChunk24);

	      resultChunk31 = _mm256_fmadd_pd(AAtemp3, BBtemp1, resultChunk31);
	      resultChunk32 = _mm256_fmadd_pd(AAtemp3, BBtemp2, resultChunk32);
	      resultChunk33 = _mm256_fmadd_pd(AAtemp3, BBtemp3, resultChunk33);
	      resultChunk34 = _mm256_fmadd_pd(AAtemp3, BBtemp4, resultChunk34);

	      resultChunk41 = _mm256_fmadd_pd(AAtemp4, BBtemp1, resultChunk41);
	      resultChunk42 = _mm256_fmadd_pd(AAtemp4, BBtemp2, resultChunk42);
	      resultChunk43 = _mm256_fmadd_pd(AAtemp4, BBtemp3, resultChunk43);
	      resultChunk44 = _mm256_fmadd_pd(AAtemp4, BBtemp4, resultChunk44);
	      #endif

	      // resultChunk21 = _mm256_fmadd_pd(AAtemp2, BBtemp1, resultChunk21);
	      // resultChunk22 = _mm256_fmadd_pd(AAtemp2, BBtemp2, resultChunk22);
	    }

	    
	    // size_t iterations = KCHUNK / 4;
	    //ii
	    //jj
	    

	    // __asm__ (
	    // 	     "movl $0, %%eax\n\t"
	    // 	     //move temporary result vector to registers
	    // 	     "vmovapd %0, %%ymm4;\n\t"
		 
	    // 	     "1:\n\t"		 
		 
	    // 	     //load components from A
	    // 	     "vmovapd (%3, %1, 8), %%ymm1;\n\t"
	    // 	     //load components from B
	    // 	     "vmovapd (%4, %2, 8), %%ymm2;\n\t"
		 
	    // 	     //multiply componentwise
	    // 	     "vmulpd %%ymm1, %%ymm2, %%ymm3;\n\t"
	    // 	     //add to temporary result vector
	    // 	     "vaddpd %%ymm3, %%ymm4, %%ymm4;\n\t"
		 
	    // 	     //calculate index + k
	    // 	     "addq $4, %1;\n\t"
	    // 	     "addq $4, %2;\n\t"
		 
	    // 	     //is k < n? loop: exit
	    // 	     "addl $4, %%eax;\n\t"
	    // 	     "cmp %5, %%eax;\n\t" // calculates k - n and stores sign as flag
	    // 	     "jl 1b;\n\t"
		 
	    // 	     //move temporary result vector back to memory
	    // 	     "vmovapd %%ymm4, %0;\n\t"
		 
	    // 	     //0, 1, 2
	    // 	     :"+m"(temp1), "+r"(indexA), "+r"(indexB)
	    // 	      //3, 4, 5
	    // 	     :"r"(A), "r"(B), "r"(n)
	    // 	     :"%ymm1", "%ymm2", "%ymm3", "%ymm4", "%eax"
	    // 	     ); 



	    avxValue converter11;
	    converter11.in = resultChunk11;

	    avxValue converter12;
	    converter12.in = resultChunk12;

	    avxValue converter13;
	    converter13.in = resultChunk13;

	    avxValue converter14;
	    converter14.in = resultChunk14;

	    avxValue converter21;
	    converter21.in = resultChunk21;

	    avxValue converter22;
	    converter22.in = resultChunk22;

	    avxValue converter23;
	    converter23.in = resultChunk23;

	    avxValue converter24;
	    converter24.in = resultChunk24;

	    avxValue converter31;
	    converter31.in = resultChunk31;

	    avxValue converter32;
	    converter32.in = resultChunk32;

	    avxValue converter33;
	    converter33.in = resultChunk33;

	    avxValue converter34;
	    converter34.in = resultChunk34;

	    avxValue converter41;
	    converter41.in = resultChunk41;

	    avxValue converter42;
	    converter42.in = resultChunk42;

	    avxValue converter43;
	    converter43.in = resultChunk43;

	    avxValue converter44;
	    converter44.in = resultChunk44;



	    // avxValue converter21;
	    // converter21.in = resultChunk21;

	    // avxValue converter22;
	    // converter22.in = resultChunk22;

	    for (size_t kk = 0; kk < 4; kk++) { //horizontal sum + single store instruction
	      result[(ii + 0) * BLOCKSIZE + jj + 0] += converter11.out[kk];
	      result[(ii + 0) * BLOCKSIZE + jj + 1] += converter12.out[kk];
	      result[(ii + 0) * BLOCKSIZE + jj + 2] += converter13.out[kk];
	      result[(ii + 0) * BLOCKSIZE + jj + 3] += converter14.out[kk];

	      result[(ii + 1) * BLOCKSIZE + jj + 0] += converter21.out[kk];
	      result[(ii + 1) * BLOCKSIZE + jj + 1] += converter22.out[kk];
	      result[(ii + 1) * BLOCKSIZE + jj + 2] += converter23.out[kk];
	      result[(ii + 1) * BLOCKSIZE + jj + 3] += converter24.out[kk];

	      result[(ii + 2) * BLOCKSIZE + jj + 0] += converter31.out[kk];
	      result[(ii + 2) * BLOCKSIZE + jj + 1] += converter32.out[kk];
	      result[(ii + 2) * BLOCKSIZE + jj + 2] += converter33.out[kk];
	      result[(ii + 2) * BLOCKSIZE + jj + 3] += converter34.out[kk];

	      result[(ii + 3) * BLOCKSIZE + jj + 0] += converter41.out[kk];
	      result[(ii + 3) * BLOCKSIZE + jj + 1] += converter42.out[kk];
	      result[(ii + 3) * BLOCKSIZE + jj + 2] += converter43.out[kk];
	      result[(ii + 3) * BLOCKSIZE + jj + 3] += converter44.out[kk];

	      // C[(i + ii + 0) * n + (j + jj + 0)] += converter11.out[kk];
	      // C[(i + ii + 0) * n + (j + jj + 1)] += converter12.out[kk];
	      // C[(i + ii + 0) * n + (j + jj + 2)] += converter13.out[kk];
	      // C[(i + ii + 0) * n + (j + jj + 3)] += converter14.out[kk];

	      // C[(i + ii + 1) * n + (j + jj + 0)] += converter21.out[kk];
	      // C[(i + ii + 1) * n + (j + jj + 1)] += converter22.out[kk];
	      // C[(i + ii + 1) * n + (j + jj + 2)] += converter23.out[kk];
	      // C[(i + ii + 1) * n + (j + jj + 3)] += converter24.out[kk];

	      // C[(i + ii + 2) * n + (j + jj + 0)] += converter31.out[kk];
	      // C[(i + ii + 2) * n + (j + jj + 1)] += converter32.out[kk];
	      // C[(i + ii + 2) * n + (j + jj + 2)] += converter33.out[kk];
	      // C[(i + ii + 2) * n + (j + jj + 3)] += converter34.out[kk];

	      // C[(i + ii + 3) * n + (j + jj + 0)] += converter41.out[kk];
	      // C[(i + ii + 3) * n + (j + jj + 1)] += converter42.out[kk];
	      // C[(i + ii + 3) * n + (j + jj + 2)] += converter43.out[kk];
	      // C[(i + ii + 3) * n + (j + jj + 3)] += converter44.out[kk];


	      // result[(ii + 1) * BLOCKSIZE + jj + 0] += converter21.out[kk];
	      // result[(ii + 1) * BLOCKSIZE + jj + 1] += converter22.out[kk];
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


/*
void multGPUASM(double *A, double *B, double *C, const uint32_t n) {

  omp_set_num_threads(2);

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

	// for (size_t k = kBlock; k < kBlock + KCHUNK; k++) {

	for (size_t ii = 0; ii < BLOCKSIZE; ii += 4) {
	  // for (size_t jj = 0; jj < BLOCKSIZE; jj++) {
	  for (size_t jj = 0; jj < BLOCKSIZE; jj += 4) {

	    register double resultChunk1[4]; //only one register and single instruction for initialization
	    register double resultChunk2[4];
	    register double resultChunk3[4];
	    register double resultChunk4[4];

	    register double resultChunk5[4]; //only one register and single instruction for initialization
	    register double resultChunk6[4];
	    register double resultChunk7[4];
	    register double resultChunk8[4];

	    register double resultChunk9[4]; //only one register and single instruction for initialization
	    register double resultChunk10[4];
	    register double resultChunk11[4];
	    register double resultChunk12[4];

	    register double resultChunk13[4]; //only one register and single instruction for initialization
	    register double resultChunk14[4];
	    register double resultChunk15[4];
	    register double resultChunk16[4];

	    for (size_t kk = 0; kk < 4; kk++) {
	      resultChunk1[kk] = 0.0;
	      resultChunk2[kk] = 0.0;
	      resultChunk3[kk] = 0.0;
	      resultChunk4[kk] = 0.0;

	      resultChunk5[kk] = 0.0;
	      resultChunk6[kk] = 0.0;
	      resultChunk7[kk] = 0.0;
	      resultChunk8[kk] = 0.0;

	      resultChunk9[kk] = 0.0;
	      resultChunk10[kk] = 0.0;
	      resultChunk11[kk] = 0.0;
	      resultChunk12[kk] = 0.0;

	      resultChunk13[kk] = 0.0;
	      resultChunk14[kk] = 0.0;
	      resultChunk15[kk] = 0.0;
	      resultChunk16[kk] = 0.0;

	    }

	    // AA + BB -> requires 8 registers
	    // resultChunk -> requires

	    for (size_t k = 0; k < KCHUNK; k += 4) {
	      for (size_t kk = 0; kk < 4; kk++) { //one SIMD lane
		resultChunk1[kk] += AA[(ii + 0) * KCHUNK + k + kk] * BB[(jj + 0) * KCHUNK + k + kk];
		resultChunk2[kk] += AA[(ii + 0) * KCHUNK + k + kk] * BB[(jj + 1) * KCHUNK + k + kk];
		resultChunk3[kk] += AA[(ii + 0) * KCHUNK + k + kk] * BB[(jj + 2) * KCHUNK + k + kk];
		resultChunk4[kk] += AA[(ii + 0) * KCHUNK + k + kk] * BB[(jj + 3) * KCHUNK + k + kk];

		resultChunk5[kk] += AA[(ii + 1) * KCHUNK + k + kk] * BB[(jj + 0) * KCHUNK + k + kk];
		resultChunk6[kk] += AA[(ii + 1) * KCHUNK + k + kk] * BB[(jj + 1) * KCHUNK + k + kk];
		resultChunk7[kk] += AA[(ii + 1) * KCHUNK + k + kk] * BB[(jj + 2) * KCHUNK + k + kk];
		resultChunk8[kk] += AA[(ii + 1) * KCHUNK + k + kk] * BB[(jj + 3) * KCHUNK + k + kk];

		resultChunk9[kk] += AA[(ii + 2) * KCHUNK + k + kk] * BB[(jj + 0) * KCHUNK + k + kk];
		resultChunk10[kk] += AA[(ii + 2) * KCHUNK + k + kk] * BB[(jj + 1) * KCHUNK + k + kk];
		resultChunk11[kk] += AA[(ii + 2) * KCHUNK + k + kk] * BB[(jj + 2) * KCHUNK + k + kk];
		resultChunk12[kk] += AA[(ii + 2) * KCHUNK + k + kk] * BB[(jj + 3) * KCHUNK + k + kk];

		resultChunk13[kk] += AA[(ii + 3) * KCHUNK + k + kk] * BB[(jj + 0) * KCHUNK + k + kk];
		resultChunk14[kk] += AA[(ii + 3) * KCHUNK + k + kk] * BB[(jj + 1) * KCHUNK + k + kk];
		resultChunk15[kk] += AA[(ii + 3) * KCHUNK + k + kk] * BB[(jj + 2) * KCHUNK + k + kk];
		resultChunk16[kk] += AA[(ii + 3) * KCHUNK + k + kk] * BB[(jj + 3) * KCHUNK + k + kk];
	      }
	      // result[ii * BLOCKSIZE + jj] += AA[ii * KCHUNK + k] * BB[jj * KCHUNK + k];
	      // result[ii * BLOCKSIZE + jj] += A[(i +  ii) * n + k] * B[(j + jj) * n + k];
	    }

	    for (size_t kk = 0; kk < 4; kk++) { //horizontal sum + single store instruction
	      result[(ii + 0) * BLOCKSIZE + jj + 0] += resultChunk1[kk];
	      result[(ii + 0) * BLOCKSIZE + jj + 1] += resultChunk2[kk];
	      result[(ii + 0) * BLOCKSIZE + jj + 2] += resultChunk3[kk];
	      result[(ii + 0) * BLOCKSIZE + jj + 3] += resultChunk4[kk];

	      result[(ii + 1) * BLOCKSIZE + jj + 0] += resultChunk5[kk];
	      result[(ii + 1) * BLOCKSIZE + jj + 1] += resultChunk6[kk];
	      result[(ii + 1) * BLOCKSIZE + jj + 2] += resultChunk7[kk];
	      result[(ii + 1) * BLOCKSIZE + jj + 3] += resultChunk8[kk];

	      result[(ii + 2) * BLOCKSIZE + jj + 0] += resultChunk9[kk];
	      result[(ii + 2) * BLOCKSIZE + jj + 1] += resultChunk10[kk];
	      result[(ii + 2) * BLOCKSIZE + jj + 2] += resultChunk11[kk];
	      result[(ii + 2) * BLOCKSIZE + jj + 3] += resultChunk12[kk];

	      result[(ii + 3) * BLOCKSIZE + jj + 0] += resultChunk13[kk];
	      result[(ii + 3) * BLOCKSIZE + jj + 1] += resultChunk14[kk];
	      result[(ii + 3) * BLOCKSIZE + jj + 2] += resultChunk15[kk];
	      result[(ii + 3) * BLOCKSIZE + jj + 3] += resultChunk16[kk];
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
*/
