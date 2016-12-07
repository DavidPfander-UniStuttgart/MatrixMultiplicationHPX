#include <cstddef>
#include <cstdint>
#include <iostream>

#include <x86intrin.h>

void multASM(double *A, double *B, double *C, const uint32_t n) {

  typedef union avxValue {
    __m256d in;
    double out[4];
  } avxValue;

  #pragma omp parallel for
  for (size_t i = 0; i < n; i += 1) {
    for (size_t j = 0; j < n; j += 1) {

      double result1 = 0.0;
      __m256d temp1 = _mm256_set1_pd(0.0);

      size_t indexA = i * n;
      size_t indexB = j * n;

#if __GNUC__ > 4 || (__GNUC__ > 4 && __GNUC_MINOR__ >= 9)
	__asm__ (
		 "movl $0, %%eax\n\t"
		 //move temporary result vector to registers
		 "vmovapd %0, %%ymm4;\n\t"
		 
		 "1:\n\t"		 
		 
		 //load components from A
		 "vmovapd (%3, %1, 8), %%ymm1;\n\t"
		 //load components from B
		 "vmovapd (%4, %2, 8), %%ymm2;\n\t"
		 
		 //multiply componentwise
		 "vmulpd %%ymm1, %%ymm2, %%ymm3;\n\t"
		 //add to temporary result vector
		 "vaddpd %%ymm3, %%ymm4, %%ymm4;\n\t"
		 
		 //calculate index + k
		 "addq $4, %1;\n\t"
		 "addq $4, %2;\n\t"
		 
		 //is k < n? loop: exit
		 "addl $4, %%eax;\n\t"
		 "cmp %5, %%eax;\n\t" // calculates k - n and stores sign as flag
		 "jl 1b;\n\t"
		 
		 //move temporary result vector back to memory
		 "vmovapd %%ymm4, %0;\n\t"
		 
		 //output: 0, 1, 2
		 :"+m"(temp1), "+r"(indexA), "+r"(indexB)
		 //input: 3, 4, 5
		 :"r"(A), "r"(B), "r"(n)
		 :"%ymm1", "%ymm2", "%ymm3", "%ymm4", "%eax"
		 ); 
	#else
	std::cout << "error: multASM requires >= gcc 4.9" << std::endl;
	#endif

      avxValue tempResult;
      
      tempResult.in = temp1;
      for (size_t i = 0; i < 4; i++) {
	result1 += tempResult.out[i];
      }

      C[i * n + j] = result1;
    }
  }
}
