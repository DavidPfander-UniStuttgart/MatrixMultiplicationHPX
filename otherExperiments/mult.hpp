

#include <vector>

/*Variants:
  naive - direct implementation
  transposed - argument B is stored transposed
  omp - adds openmp
  vectorized - enables vectorization
  intrinsics - vectorization by intrinsics
  asm - vectorization by assembly

 */

void multNaive(double *A, double *B, double *C, const uint32_t n);

void multTransposed(double *A, double *B, double *C, const uint32_t n);

void multCounter(double *A, double *B, double *C, const uint32_t n);

void multExchanged(double *A, double *B, double *C, const uint32_t n);

void multRegister(double *A, double *B, double *C, const uint32_t n);

void multIntrinLoop(double *A, double *B, double *C, const uint32_t n);

// void multRegBlock(double *A, double *B, double *C, const uint32_t n);

//void multCounterOMP(double *A, double *B, double *C, const uint32_t n);

void multBlocked2(double *A, double *B, double *C, const uint32_t n);

void multBlocked4(double *A, double *B, double *C, const uint32_t n);

void multGPU(double *A, double *B, double *C, const uint32_t n);

void multGPUASM(double *A, double *B, double *C, const uint32_t n);

void multASM(double *A, double *B, double *C, const uint32_t n);

//void mult(double *__restrict__ A, double *__restrict__ B, double *__restrict__ C, const uint32_t n);

//void mult(std::vector<double> &A, std::vector<double> &B, std::vector<double> &C, const size_t n);
