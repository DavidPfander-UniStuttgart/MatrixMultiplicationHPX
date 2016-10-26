/*
 * matrix_multiply_algorithm.hpp
 *
 *  Created on: Oct 10, 2016
 *      Author: pfandedd
 */

#include "matrix_multiply_kernel_test.hpp"

#include <Vc/Vc>
#include <boost/align/aligned_allocator.hpp>

#define PADDING 64
#define L3_X 256 // max 2 L3 par set to 1024 (rest 512)
#define L3_Y 256
#define L3_K_STEP 512
#define L2_X 128 // max 2 L2 par set to 128 (rest 64)
#define L2_Y 128
#define L2_K_STEP 64
#define L1_X 64 // max all L1 par set to 32
#define L1_Y 32
#define L1_K_STEP 16
#define X_REG 5
#define Y_REG 8

namespace kernel_test {

  matrix_multiply_kernel_test::matrix_multiply_kernel_test(size_t N, std::vector<double> &A,
							   std::vector<double> &B, bool transposed,
							   uint64_t repetitions, uint64_t verbose) :
    N(N), A(A), B(B), repetitions(repetitions), verbose(verbose) {

  }

  std::vector<double> matrix_multiply_kernel_test::matrix_multiply() {

    std::vector<double> C(N * N);
    std::fill(C.begin(), C.end(), 0.0);
      
    std::vector<double, boost::alignment::aligned_allocator<double, 32>> C_padded((N + PADDING) * (N + PADDING));
    std::fill(C_padded.begin(), C_padded.end(), 0.0);

    std::vector<double, boost::alignment::aligned_allocator<double, 32>> A_padded((N + PADDING) * (N + PADDING));
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
	A_padded[i * (N + PADDING) + j] = A[i * N + j];	    
      }	
    }

    // is also padded if padding is enabled
    std::vector<double, boost::alignment::aligned_allocator<double, 32>> A_trans((N + PADDING) * (N + PADDING));
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
	A_trans[i * (N + PADDING) + j] = A[j * N + i];	    
      }	
    }

    std::vector<double, boost::alignment::aligned_allocator<double, 32>> B_padded((N + PADDING) * (N + PADDING));
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
	B_padded[i * (N + PADDING) + j] = B[i * N + j];	    
      }	
    }	

    for (size_t rep = 0; rep < repetitions; rep++) {
	  
      using Vc::double_v;
      // L3 blocking
#pragma omp parallel for collapse(2)
      for (size_t l3_x = 0; l3_x < N; l3_x += L3_X) {
	for (size_t l3_y = 0; l3_y < N; l3_y += L3_Y) {
	  for (size_t l3_k = 0; l3_k < N; l3_k += L3_K_STEP) {
	    // L2 blocking
	    for (size_t l2_x = l3_x; l2_x < l3_x + L3_X; l2_x += L2_X) {
	      for (size_t l2_y = l3_y; l2_y < l3_y + L3_Y; l2_y += L2_Y) {
		for (size_t l2_k = l3_k; l2_k < l3_k + L3_K_STEP; l2_k += L2_K_STEP) {
		  // L1 blocking
		  for (size_t l1_x = l2_x; l1_x < l2_x + L2_X; l1_x += L1_X) {
		    for (size_t l1_y = l2_y; l1_y < l2_y + L2_Y; l1_y += L1_Y) {
		      for (size_t l1_k = l2_k; l1_k < l2_k + L2_K_STEP; l1_k += L1_K_STEP) {
			// Register blocking
			for (size_t x = l1_x + 0; x < l1_x + L1_X; x += X_REG) {
			  for (size_t y = l1_y + 0; y < l1_y + L1_Y; y += Y_REG) {

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

			    double_v b_temp_1 = double_v(&B_padded[l1_k * (N + PADDING) + y], Vc::Aligned);
			    double_v b_temp_2 = double_v(&B_padded[l1_k * (N + PADDING) + (y + 4)], Vc::Aligned);
	      
			    for (size_t k_inner = l1_k; k_inner < l1_k + L1_K_STEP; k_inner += 1) {

			      double_v a_temp_1 = A_trans[k_inner * (N + PADDING) + (x + 0)];
			      double_v a_temp_2 = A_trans[k_inner * (N + PADDING) + (x + 1)];
			      double_v a_temp_3 = A_trans[k_inner * (N + PADDING) + (x + 2)];

			      acc_11 += a_temp_1 * b_temp_1;
			      acc_21 += a_temp_2 * b_temp_1;
		
			      acc_12 += a_temp_1 * b_temp_2;
			      acc_22 += a_temp_2 * b_temp_2;

			      double_v a_temp_4 = A_trans[k_inner * (N + PADDING) + (x + 3)];
			      double_v a_temp_5 = A_trans[k_inner * (N + PADDING) + (x + 4)];

			      acc_31 += a_temp_3 * b_temp_1;		
			      acc_41 += a_temp_4 * b_temp_1;
			      acc_51 += a_temp_5 * b_temp_1;

			      b_temp_1 = double_v(&B_padded[(k_inner + 1) * (N + PADDING) + y], Vc::Aligned);
		
			      acc_42 += a_temp_4 * b_temp_2;
			      acc_52 += a_temp_5 * b_temp_2;
			      acc_32 += a_temp_3 * b_temp_2;

			      b_temp_2 = double_v(&B_padded[(k_inner + 1) * (N + PADDING) + (y + 4)], Vc::Aligned);
			    }
		    
			    double_v res_11 = double_v(&C_padded[(x + 0) * (N + PADDING) + y]);
			    res_11 += acc_11;
			    res_11.store(&C_padded[(x + 0) * (N + PADDING) + y]);
			    if (x + 1 < l1_x + L1_X) {
			      double_v res_21 = double_v(&C_padded[(x + 1) * (N + PADDING) + y]);
			      res_21 += acc_21;
			      res_21.store(&C_padded[(x + 1) * (N + PADDING) + y]);
			    }
			    if (x + 2 < l1_x + L1_X) {
			      double_v res_31 = double_v(&C_padded[(x + 2) * (N + PADDING) + y]);
			      res_31 += acc_31;
			      res_31.store(&C_padded[(x + 2) * (N + PADDING) + y]);
			    }
			    if (x + 3 < l1_x + L1_X) {
			      double_v res_41 = double_v(&C_padded[(x + 3) * (N + PADDING) + y]);
			      res_41 += acc_41;
			      res_41.store(&C_padded[(x + 3) * (N + PADDING) + y]);
			    }
		    
			    if (x + 4 < l1_x + L1_X) {
			      double_v res_51 = double_v(&C_padded[(x + 4) * (N + PADDING) + y]);
			      res_51 += acc_51;
			      res_51.store(&C_padded[(x + 4) * (N + PADDING) + y]);
			    }

			    double_v res_12 = double_v(&C_padded[(x + 0) * (N + PADDING) + (y + 4)]);
			    res_12 += acc_12;
			    res_12.store(&C_padded[(x + 0) * (N + PADDING) + (y + 4)]);
			    if (x + 1 < l1_x + L1_X) {
			      double_v res_22 = double_v(&C_padded[(x + 1) * (N + PADDING) + (y + 4)]);
			      res_22 += acc_22;
			      res_22.store(&C_padded[(x + 1) * (N + PADDING) + (y + 4)]);
			    }
			    if (x + 2 < l1_x + L1_X) {
			      double_v res_32 = double_v(&C_padded[(x + 2) * (N + PADDING) + (y + 4)]);
			      res_32 += acc_32;
			      res_32.store(&C_padded[(x + 2) * (N + PADDING) + (y + 4)]);
			    }
			    if (x + 3 < l1_x + L1_X) {
			      double_v res_42 = double_v(&C_padded[(x + 3) * (N + PADDING) + (y + 4)]);
			      res_42 += acc_42;
			      res_42.store(&C_padded[(x + 3) * (N + PADDING) + (y + 4)]);
			    }
			    if (x + 4 < l1_x + L1_X) {
			      double_v res_52 = double_v(&C_padded[(x + 4) * (N + PADDING) + (y + 4)]);
			      res_52 += acc_52;
			      res_52.store(&C_padded[(x + 4) * (N + PADDING) + (y + 4)]);
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

    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
	C[i * N + j] = C_padded[i * (N + PADDING) + j];	    
      }	
    }	
	
    return C;
  }
}

#undef PADDING
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
