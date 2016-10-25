/*
 * matrix_multiply_algorithm.hpp
 *
 *  Created on: Oct 10, 2016
 *      Author: pfandedd
 */

#pragma once

#include <boost/iterator/iterator_facade.hpp>
#include "hpx/util/iterator_facade.hpp"
#include <Vc/Vc>
#include <boost/align/aligned_allocator.hpp>

// all blocking specifications have to divide N (except X_REG, Y_REG)
#define PADDING_X 64
#define L3_X 256 // max 2 L3 par set to 1024 (rest 512)
#define L3_Y 256
#define L3_K_STEP 256
#define L2_X 128 // max 2 L2 par set to 128 (rest 64)
#define L2_Y 128
#define L2_K_STEP 64
#define L1_X 64 // max all L1 par set to 32
#define L1_Y 32
#define L1_K_STEP 16
#define X_REG 5 // cannot be changed!
#define Y_REG 8 // cannot be changed!

namespace kernel_tiled {

class matrix_multiply_kernel_tiled {

private:
    size_t N;
    std::vector<double> &A;
    std::vector<double> &B;

    uint64_t repetitions;
    uint64_t verbose;
public:
    matrix_multiply_kernel_tiled(size_t N, std::vector<double> &A,
				 std::vector<double> &B, bool transposed,
				 uint64_t repetitions, uint64_t verbose) :
      N(N), A(A), B(B), repetitions(repetitions), verbose(verbose) {

    }

  std::vector<double> matrix_multiply() {

    std::vector<double> C(N * N);
    std::fill(C.begin(), C.end(), 0.0);

    // create a matrix of l1 cachable submatrices, caching by tiling, no large strides even without padding
    std::vector<double, boost::alignment::aligned_allocator<double, 32>> C_padded((N + PADDING_X) * N);
    std::fill(C_padded.begin(), C_padded.end(), 0.0);

    // create a matrix of l1 cachable submatrices, caching by tiling, no large strides even without padding
    // is also padded if padding is enabled (row padded only)
    std::vector<double, boost::alignment::aligned_allocator<double, 32>> A_trans((N + PADDING_X) * N);
    for (size_t l1_x = 0; l1_x < N/L1_X; l1_x += 1) {
      for (size_t l1_k = 0; l1_k < N/L1_K_STEP; l1_k += 1) {
	size_t base_index = (L1_X * L1_K_STEP) * (l1_k * (N/L1_X) + l1_x); // look up submatrix
	for (size_t x = 0; x < L1_X; x++) {
	  for (size_t k = 0; k < L1_K_STEP; k++) {
	    A_trans[base_index + k * L1_X + x] = A[(l1_x * L1_X + x) * N + (l1_k * L1_K_STEP + k)];
	  }
	}
      }
    }

    // don't need padding for B, no dependency to row count
    std::vector<double, boost::alignment::aligned_allocator<double, 32>> B_padded(N * N);
    for (size_t l1_y = 0; l1_y < (N/L1_Y); l1_y += 1) {
      for (size_t l1_k = 0; l1_k < (N/L1_K_STEP); l1_k += 1) {
	size_t base_index = (L1_Y * L1_K_STEP) * (l1_k * (N/L1_Y) + l1_y); // look up submatrix
	for (size_t y = 0; y < L1_Y; y++) {
	  for (size_t k = 0; k < L1_K_STEP; k++) {
	    B_padded[base_index + k * L1_Y + y] = B[(l1_k * L1_K_STEP + k) * N + (l1_y * L1_Y + y)];
	  }
	}
      }
    }

    //TODO: test with padding enabled!
    
    // std::cout << "A:" << std::endl;
    // for (uint64_t x = 0; x < N; x++) {
    //   for (uint64_t y = 0; y < N; y++) {
    // 	if (y > 0) {
    // 	  std::cout << ", ";
    // 	}
    // 	std::cout << A[x * N + y];
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "A trans:" << std::endl;
    // for (uint64_t x = 0; x < N; x++) {
    //   for (uint64_t y = 0; y < N; y++) {
    // 	if (y > 0) {
    // 	  std::cout << ", ";
    // 	}
    // 	std::cout << A[y * N + x];
    //   }
    //   std::cout << std::endl;
    // }

    // std::cout << "A_trans (sub-matrices, DOESNT CONSIDER PADDING):" << std::endl;
    // size_t submatrix_index = 0;
    // for (uint64_t k = 0; k < N * N; k++) {      
    //   if (submatrix_index < L1_X * L1_K_STEP) {
    // 	if (submatrix_index < L1_X * L1_K_STEP) {
    // 	  if (submatrix_index > 0) {
    // 	    std::cout << ", ";
    // 	  }
    // 	  std::cout << A_trans[k];
    // 	}
    // 	submatrix_index++;
    //   } else {	
    // 	submatrix_index = 1;
    // 	std::cout << std::endl;
    // 	std::cout << A_trans[k];
    //   }
    // }
    // std::cout << std::endl;
    
    // std::cout << "B:" << std::endl;
    // for (uint64_t x = 0; x < N; x++) {
    //   for (uint64_t y = 0; y < N; y++) {
    // 	if (y > 0) {
    // 	  std::cout << ", ";
    // 	}
    // 	std::cout << B[x * N + y];
    //   }
    //   std::cout << std::endl;
    // }

    // std::cout << "B_padded (sub-matrices, DOESNT CONSIDER PADDING):" << std::endl;
    // submatrix_index = 0;
    // for (uint64_t k = 0; k < N * N; k++) {      
    //   if (submatrix_index < L1_Y * L1_K_STEP) {
    // 	if (submatrix_index < L1_Y * L1_K_STEP) {
    // 	  if (submatrix_index > 0) {
    // 	    std::cout << ", ";
    // 	  }
    // 	  std::cout << B_padded[k];
    // 	}
    // 	submatrix_index++;
    //   } else {	
    // 	submatrix_index = 1;
    // 	std::cout << std::endl;
    // 	std::cout << B_padded[k];
    //   }
    // }
    // std::cout << std::endl;
    
    for (size_t rep = 0; rep < repetitions; rep++) {
	  
      using Vc::double_v;
      // L3 blocking
// #pragma omp parallel for collapse(2)
      for (size_t l3_x = 0; l3_x < N; l3_x += L3_X) {
	for (size_t l3_y = 0; l3_y < N; l3_y += L3_Y) {
	  for (size_t l3_k = 0; l3_k < N; l3_k += L3_K_STEP) {
	    // L2 blocking
	    for (size_t l2_x = l3_x; l2_x < l3_x + L3_X; l2_x += L2_X) {
	      for (size_t l2_y = l3_y; l2_y < l3_y + L3_Y; l2_y += L2_Y) {
		for (size_t l2_k = l3_k; l2_k < l3_k + L3_K_STEP; l2_k += L2_K_STEP) {
		  // L1 blocking
		  for (size_t l1_x = l2_x; l1_x < l2_x + L2_X; l1_x += L1_X) {
		    size_t l1_block_x = l1_x / L1_X;
		    for (size_t l1_y = l2_y; l1_y < l2_y + L2_Y; l1_y += L1_Y) {
		      size_t l1_block_y = l1_y / L1_Y;
		      size_t C_base_index = (L1_X * L1_Y) * (l1_block_x * (N/L1_Y) + l1_block_y);
		      for (size_t l1_k = l2_k; l1_k < l2_k + L2_K_STEP; l1_k += L1_K_STEP) {
			size_t l1_block_k = l1_k / L1_K_STEP;
			size_t A_base_index = (L1_X * L1_K_STEP) * (l1_block_k * (N/L1_X) + l1_block_x);
			size_t B_base_index = (L1_Y * L1_K_STEP) * (l1_block_k * (N/L1_Y) + l1_block_y);
			// Register blocking
			for (size_t x = 0; x < L1_X; x += X_REG) {
			  for (size_t y = 0; y < L1_Y; y += Y_REG) {

			    // double_v acc = double_v(C[x * N + y]);
			    double_v acc_11 = 0.0;
			    double_v acc_21 = 0.0;
			    double_v acc_31 = 0.0;
			    double_v acc_41 = 0.0;

			    double_v acc_51 = 0.0;
	      
			    double_v acc_12 = 0.0;
			    double_v acc_22 = 0.0;
			    double_v acc_32 = 0.0;
			    double_v acc_42 = 0.0;

			    double_v acc_52 = 0.0;

			    double_v b_temp_1 =
			      double_v(&B_padded[B_base_index + 0 * L1_Y + y], Vc::Aligned);
			    double_v b_temp_2 =
			      double_v(&B_padded[B_base_index + 0 * L1_Y + (y + 4)], Vc::Aligned);
	      
			    for (size_t k_inner = 0; k_inner < L1_K_STEP; k_inner += 1) {

			      double_v a_temp_1 = A_trans[A_base_index + k_inner * L1_X + (x + 0)];
			      double_v a_temp_2 = A_trans[A_base_index + k_inner * L1_X + (x + 1)];
			      double_v a_temp_3 = A_trans[A_base_index + k_inner * L1_X + (x + 2)];

			      acc_11 += a_temp_1 * b_temp_1;
			      acc_21 += a_temp_2 * b_temp_1;
		
			      acc_12 += a_temp_1 * b_temp_2;
			      acc_22 += a_temp_2 * b_temp_2;

			      double_v a_temp_4 = A_trans[A_base_index + k_inner * L1_X + (x + 3)];
			      double_v a_temp_5 = A_trans[A_base_index + k_inner * L1_X + (x + 4)];

			      acc_31 += a_temp_3 * b_temp_1;		
			      acc_41 += a_temp_4 * b_temp_1;
			      acc_51 += a_temp_5 * b_temp_1;

			      b_temp_1 =
				double_v(&B_padded[B_base_index + (k_inner + 1) * L1_Y + y], Vc::Aligned);
		
			      acc_42 += a_temp_4 * b_temp_2;
			      acc_52 += a_temp_5 * b_temp_2;
			      acc_32 += a_temp_3 * b_temp_2;

			      b_temp_2 =
				double_v(&B_padded[B_base_index + (k_inner + 1) * L1_Y + (y + 4)], Vc::Aligned);
			    }
		    
			    double_v res_11 = double_v(&C_padded[C_base_index + (x + 0) * L1_Y + y]);
			    res_11 += acc_11;
			    res_11.store(&C_padded[C_base_index + (x + 0) * L1_Y + y]);
			    if (x + 1 < L1_X) {
			      double_v res_21 = double_v(&C_padded[C_base_index + (x + 1) * L1_Y + y]);
			      res_21 += acc_21;
			      res_21.store(&C_padded[C_base_index + (x + 1) * L1_Y + y]);
			    }
			    if (x + 2 < L1_X) {
			      double_v res_31 = double_v(&C_padded[C_base_index + (x + 2) * L1_Y + y]);
			      res_31 += acc_31;
			      res_31.store(&C_padded[C_base_index + (x + 2) * L1_Y + y]);
			    }
			    if (x + 3 < L1_X) {
			      double_v res_41 = double_v(&C_padded[C_base_index + (x + 3) * L1_Y + y]);
			      res_41 += acc_41;
			      res_41.store(&C_padded[C_base_index + (x + 3) * L1_Y + y]);
			    }
		    
			    if (x + 4 < L1_X) {
			      double_v res_51 = double_v(&C_padded[C_base_index + (x + 4) * L1_Y + y]);
			      res_51 += acc_51;
			      res_51.store(&C_padded[C_base_index + (x + 4) * L1_Y + y]);
			    }

			    if (y + 4 < L1_Y) {
			      double_v res_12 = double_v(&C_padded[C_base_index + (x + 0) * L1_Y + (y + 4)]);
			      res_12 += acc_12;
			      res_12.store(&C_padded[C_base_index + (x + 0) * L1_Y + (y + 4)]);
			    }
			    if (x + 1 < L1_X) {
			      double_v res_22 = double_v(&C_padded[C_base_index + (x + 1) * L1_Y + (y + 4)]);
			      res_22 += acc_22;
			      res_22.store(&C_padded[C_base_index + (x + 1) * L1_Y + (y + 4)]);
			    }
			    if (x + 2 < L1_X) {
			      double_v res_32 = double_v(&C_padded[C_base_index + (x + 2) * L1_Y + (y + 4)]);
			      res_32 += acc_32;
			      res_32.store(&C_padded[C_base_index + (x + 2) * L1_Y + (y + 4)]);
			    }
			    if (x + 3 < L1_X) {
			      double_v res_42 = double_v(&C_padded[C_base_index + (x + 3) * L1_Y + (y + 4)]);
			      res_42 += acc_42;
			      res_42.store(&C_padded[C_base_index + (x + 3) * L1_Y + (y + 4)]);
			    }
			    if (x + 4 < L1_X) {
			      double_v res_52 = double_v(&C_padded[C_base_index + (x + 4) * L1_Y + (y + 4)]);
			      res_52 += acc_52;
			      res_52.store(&C_padded[C_base_index + (x + 4) * L1_Y + (y + 4)]);
			    }
			  }
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }

    // std::cout << "C_padded result (sub-matrices, DOESNT CONSIDER PADDING):" << std::endl;
    // submatrix_index = 0;
    // for (uint64_t k = 0; k < N * N; k++) {      
    //   if (submatrix_index < L1_Y * L1_X) {
    // 	if (submatrix_index < L1_Y * L1_X) {
    // 	  if (submatrix_index > 0) {
    // 	    std::cout << ", ";
    // 	  }
    // 	  std::cout << C_padded[k];
    // 	}
    // 	submatrix_index++;
    //   } else {	
    // 	submatrix_index = 1;
    // 	std::cout << std::endl;
    // 	std::cout << C_padded[k];
    //   }
    // }
    // std::cout << std::endl;
    
    for (size_t l1_x = 0; l1_x < N/L1_X; l1_x += 1) {
      for (size_t l1_y = 0; l1_y < N/L1_Y; l1_y += 1) {
	size_t base_index = (L1_X * L1_Y) * (l1_x * (N/L1_Y) + l1_y); // look up submatrix
    	for (size_t x = 0; x < L1_X; x++) {
    	  for (size_t y = 0; y < L1_Y; y++) {
    	    C.at((l1_x * L1_X + x) * N + (l1_y * L1_Y + y)) = C_padded.at(base_index + x * L1_Y + y);
    	  }
    	}
      }
    }
	
    return C;
  }
};

}

#undef PADDING_X
#undef L3_X
#undef L3_Y
#undef L3_K_STEP
#undef L2_X
#undef L2_Y
#undef L2_K_STEP
#undef L1_X
#undef L1_Y
#undef L1_K_STEP
#undef X_REG
#undef Y_REG
