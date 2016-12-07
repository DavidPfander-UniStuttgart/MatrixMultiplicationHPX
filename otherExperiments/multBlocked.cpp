#include <cstddef>
#include <cstdint>

void multBlocked2(double *A, double *B, double *C, const uint32_t n) {

  // if (n == 0) {
  //   return;
  // }

  // A = (double *) __builtin_assume_aligned(A, 64);
  // B = (double *) __builtin_assume_aligned(B, 64);
  // C = (double *) __builtin_assume_aligned(C, 64);

  #pragma omp parallel for
  //REMARK: blocking with blocksize 2 -> ~2x performance
  for (size_t i = 0; i < n; i += 2) {
    for (size_t j = 0; j < n; j += 2) {

      //REMARK: g++-4.9 doesn't report successfully vectorized loop
      double result1 = 0.0;
      double result2 = 0.0;
      double result3 = 0.0;
      double result4 = 0.0;

      //REMARK: mixing 32-bit int and 64-bit int -> no vectorize (clang && gcc)
      //#pragma clang loop vectorize(enable)

      //REMARK: loop step != 1, loopd doesn't vectorize (gcc)
      for (size_t k = 0; k < n; k += 1) {
	//REMARK: does vectorize with clang (and with gcc (but unclear output with gcc))
	result1 += A[i * n + k] * B[j * n + k];

	result2 += A[(i + 1) * n + k] * B[j * n + k];
	result3 += A[i * n + k] * B[(j + 1) * n + k];
	result4 += A[(i + 1) * n + k] * B[(j + 1) * n + k];
      }
      C[i * n + j] = result1;
      C[(i + 1) * n + j] = result2;
      C[i * n + (j + 1)] = result3;
      C[(i + 1) * n + (j + 1)] = result4;
    }
  }
}



void multBlocked4(double *A, double *B, double *C, const uint32_t n) {

  // if (n == 0) {
  //   return;
  // }

  // A = (double *) __builtin_assume_aligned(A, 64);
  // B = (double *) __builtin_assume_aligned(B, 64);
  // C = (double *) __builtin_assume_aligned(C, 64);


  #pragma omp parallel for
  //REMARK: blocking with blocksize 2 -> ~2x performance
  for (size_t i = 0; i < n; i += 4) {
    for (size_t j = 0; j < n; j += 4) {

      //REMARK: g++-4.9 doesn't report successfully vectorized loop
      double result1 = 0.0;
      double result2 = 0.0;
      double result3 = 0.0;
      double result4 = 0.0;

      double result5 = 0.0;
      double result6 = 0.0;
      double result7 = 0.0;
      double result8 = 0.0;

      double result9 = 0.0;
      double result10 = 0.0;
      double result11 = 0.0;
      double result12 = 0.0;

      double result13 = 0.0;
      double result14 = 0.0;
      double result15 = 0.0;
      double result16 = 0.0;
      //REMARK: mixing 32-bit int and 64-bit int -> no vectorize (clang && gcc)
      //#pragma clang loop vectorize(enable)

      //REMARK: loop step != 1, loopd doesn't vectorize (gcc)
      for (size_t k = 0; k < n; k += 4) {
	//REMARK: does vectorize with clang (and with gcc (but unclear output with gcc))
	// result1 += A[i * n + k] * B[j * n + k];
	// result2 += A[(i + 1) * n + k] * B[j * n + k];
	// result3 += A[i * n + k] * B[(j + 1) * n + k];
	// result4 += A[(i + 1) * n + k] * B[(j + 1) * n + k];

	result1 += A[i * n + k] * B[j * n + k];
	result2 += A[i * n + k] * B[(j + 1) * n + k];
	result3 += A[i * n + k] * B[(j + 2) * n + k];
	result4 += A[i * n + k] * B[(j + 3) * n + k];

	result5 += A[(i + 1) * n + k] * B[j * n + k];
	result6 += A[(i + 1) * n + k] * B[(j + 1) * n + k];
	result7 += A[(i + 1) * n + k] * B[(j + 2) * n + k];
	result8 += A[(i + 1) * n + k] * B[(j + 3) * n + k];

	result9 += A[(i + 2) * n + k] * B[j * n + k];
	result10 += A[(i + 2) * n + k] * B[(j + 1) * n + k];
	result11 += A[(i + 2) * n + k] * B[(j + 2) * n + k];
	result12 += A[(i + 2) * n + k] * B[(j + 3) * n + k];

	result13 += A[(i + 3) * n + k] * B[j * n + k];
	result14 += A[(i + 3) * n + k] * B[(j + 1) * n + k];
	result15 += A[(i + 3) * n + k] * B[(j + 2) * n + k];
	result16 += A[(i + 3) * n + k] * B[(j + 3) * n + k];

	//REMARK: doesn't vectorize with clang
	//C[i * n + j] += A[i * n + k] * B[j * n + k];


	// C[i * n + j] += A[i * n + k + 1] * B[j * n + k + 1];
	// C[i * n + j] += A[i * n + k + 2] * B[j * n + k + 2];
	// C[i * n + j] += A[i * n + k + 3] * B[j * n + k + 3];
      }
      // C[i * n + j] = result1;
      // C[(i + 1) * n + j] = result2;
      // C[i * n + (j + 1)] = result3;
      // C[(i + 1) * n + (j + 1)] = result4;

      C[i * n + j] = result1;
      C[i * n + (j + 1)] = result2;
      C[i * n + (j + 2)] = result3;
      C[i * n + (j + 3)] = result4;

      C[(i + 1) * n + j] = result5;
      C[(i + 1) * n + (j + 1)] = result6;
      C[(i + 1) * n + (j + 2)] = result7;
      C[(i + 1) * n + (j + 3)] = result8;

      C[(i + 2) * n + j] = result9;
      C[(i + 2) * n + (j + 1)] = result10;
      C[(i + 2) * n + (j + 2)] = result11;
      C[(i + 2) * n + (j + 3)] = result12;

      C[(i + 3) * n + j] = result13;
      C[(i + 3) * n + (j + 1)] = result14;
      C[(i + 3) * n + (j + 2)] = result15;
      C[(i + 3) * n + (j + 3)] = result16;
    }
  }
}
