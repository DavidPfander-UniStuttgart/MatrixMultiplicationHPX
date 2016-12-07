#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>
#include <iomanip>

#include "mult.hpp"

inline void *align( size_t alignment, size_t size,
                    void *&ptr, std::size_t space ) {
	std::uintptr_t pn = reinterpret_cast< std::uintptr_t >( ptr );
	std::uintptr_t aligned = ( pn + alignment - 1 ) & - alignment;
	std::size_t padding = aligned - pn;
	if ( space < size + padding ) return nullptr;
	space -= padding;
	return ptr = reinterpret_cast< void * >( aligned );
}

int main (int argc, char **argv) {

  const size_t n = 4096; //make iterations const -> g++ can detect loop

  double *AUnaligned = new double[n * n + 8];
  double *ATransposedUnaligned = new double[n * n + 8];
  double *BUnaligned = new double[n * n + 8];
  double *BTransposedUnaligned = new double[n * n + 8];
  double *CUnaligned = new double[n * n + 8];
  
  //void *AVoid = static_cast<void *>(AUnaligned);
  double *A = static_cast<double *>(align(64, sizeof(double), (void *&) AUnaligned, n * n + 8));

  double *ATransposed = static_cast<double *>(align(64, sizeof(double), (void *&) ATransposedUnaligned, n * n + 8));

  double *B = static_cast<double *>(align(64, sizeof(double), (void *&) BUnaligned, n * n + 8));

  double *BTransposed = static_cast<double *>(align(64, sizeof(double), (void *&) BTransposedUnaligned, n * n + 8));

  double *C = static_cast<double *>(align(64, sizeof(double), (void *&) CUnaligned, n * n + 8));

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      A[i * n + j] = 2.0;
      ATransposed[j * n + i] = 2.0;
      if (i == j) {
	B[i * n + j] = 1.0; //initialize B
      } else {
	B[i * n + j] = 0.0; //initialize B
      }

      if (i == j) {
      	BTransposed[j * n + i] = 1.0; //initialize B transposed
      } else {
      	BTransposed[j * n + i] = 0.0; //initialize B transposed
      }
      // BTransposed[j * n + i] = static_cast<double>(i);

      C[i * n + j] = 0.0;
    }
  }

  size_t iter = 1;
  double flops = iter * 2 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
  // double flops = 2 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
  std::cout << "total flops 2n^3 -> " << std::setprecision(12) << flops << std::endl;

  if (argc == 1) {
    std::cout << "no implementation specified: exiting" << std::endl;
    return 0;
  }

  auto startTime = std::chrono::high_resolution_clock::now();

  if (strcmp(argv[1], "naive") == 0) {
    for (size_t i = 0; i < iter; i++) {
      multNaive(A, B, C, n);
    }
  } else if (strcmp(argv[1], "transposed") == 0) {
    for (size_t i = 0; i < iter; i++) {
      multTransposed(A, BTransposed, C, n);
    }
  } else if (strcmp(argv[1], "counter") == 0) { // also is vectorized (!)
    for (size_t i = 0; i < iter; i++) {
      multCounter(A, BTransposed, C, n);
    }
  } else if (strcmp(argv[1], "exchanged") == 0) { // also is vectorized (!)
    for (size_t i = 0; i < iter; i++) {
      //not transposed!
      multExchanged(A, B, C, n);
    }
  } else if (strcmp(argv[1], "blocked2") == 0) {
    for (size_t i = 0; i < iter; i++) {
      multBlocked2(A, BTransposed, C, n);
    }
  } else if (strcmp(argv[1], "blocked4") == 0) {
    for (size_t i = 0; i < iter; i++) {
      multBlocked4(A, BTransposed, C, n);
    }
  } else if (strcmp(argv[1], "asm") == 0) {
    for (size_t i = 0; i < iter; i++) {
      multASM(A, BTransposed, C, n);
    }
  } else if (strcmp(argv[1], "gpu") == 0) {
    for (size_t i = 0; i < iter; i++) {
      multGPU(A, BTransposed, C, n);
    }
  } else if (strcmp(argv[1], "gpuasm") == 0) {
    for (size_t i = 0; i < iter; i++) {
      multGPUASM(A, BTransposed, C, n);
    }
  } else if (strcmp(argv[1], "register") == 0) {
    for (size_t i = 0; i < iter; i++) {
      multRegister(A, B, C, n);
    }
  } else if (strcmp(argv[1], "intrinLoop") == 0) {
    for (size_t i = 0; i < iter; i++) {
      multIntrinLoop(A, B, C, n);
      // multIntrinLoop(A, BTransposed, C, n);
      // multIntrinLoop(ATransposed, B, C, n);
            }
  // } else if (strcmp(argv[1], "regBlock") == 0) {
  //   for (size_t i = 0; i < iter; i++) {
  //     multRegBlock(A, B, C, n);
  //     // multIntrinLoop(A, BTransposed, C, n);
  //     // multIntrinLoop(ATransposed, B, C, n);
  //   }
  } else {
    std::cout << "unknown implementation specified: exiting" << std::endl;
    return 0;
  }


  auto stopTime = std::chrono::high_resolution_clock::now();
  auto duration = stopTime - startTime;
  double seconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0;
  double gflop = flops / 1E9;
  std::cout << "gflop: " << gflop << std::endl;
  std::cout << "seconds: " << seconds << std::endl;

  std::cout << (gflop / seconds) << " Gflops" << std::endl;
  

  double check = 0;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      check += C[i * n + j] * C[i * n + j];
    }
  }

  check = sqrt(check);

  std::cout << "check: " << check << std::endl;

  return 0;
}

