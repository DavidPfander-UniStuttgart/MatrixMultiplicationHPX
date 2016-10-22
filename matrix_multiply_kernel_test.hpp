/*
 * matrix_multiply_algorithm.hpp
 *
 *  Created on: Oct 10, 2016
 *      Author: pfandedd
 */

#pragma once

#include "hpx/parallel/algorithms/for_each.hpp"
#include "hpx/parallel/algorithms/for_loop.hpp"
#include "hpx/parallel/execution_policy.hpp"
#include <boost/iterator/iterator_facade.hpp>
#include "hpx/util/iterator_facade.hpp"
#include <Vc/Vc>
#include <boost/align/aligned_allocator.hpp>

namespace kernel_test {

class matrix_multiply_kernel_test {

private:
    size_t N;
    std::vector<double> &A;
    std::vector<double> &B;
    bool transposed;

    uint64_t block_result;
    uint64_t block_input;
    uint64_t repetitions;
    uint64_t verbose;
public:
    matrix_multiply_kernel_test(size_t N, std::vector<double> &A,
            std::vector<double> &B, bool transposed, uint64_t block_result,
            uint64_t block_input, uint64_t repetitions, uint64_t verbose) :
            N(N), A(A), B(B), transposed(transposed), block_result(
                    block_result), block_input(block_input), repetitions(
                    repetitions), verbose(verbose) {

    }

    std::vector<double> matrix_multiply() {

      #define PADDING 8
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
//      for (size_t x = 0; x < N; x += 4) {
//        for (size_t y = 0; y < N; y += 2) {
////          for (size_t k = 0; k < N; k += block_input) {
//          double result_component_0_0 = 0.0;
//          double result_component_0_1 = 0.0;
//          double result_component_1_0 = 0.0;
//          double result_component_1_1 = 0.0;
//
//          double result_component_2_0 = 0.0;
//          double result_component_2_1 = 0.0;
//          double result_component_3_0 = 0.0;
//          double result_component_3_1 = 0.0;
//
////          double result_component_0_2 = 0.0;
////          double result_component_0_3 = 0.0;
////          double result_component_1_2 = 0.0;
////          double result_component_1_3 = 0.0;
////          double result_component_2_2 = 0.0;
////          double result_component_2_3 = 0.0;
////          double result_component_3_2 = 0.0;
////          double result_component_3_3 = 0.0;
//
//          for (size_t k_inner = 0; k_inner < N; k_inner++) {
//
//            result_component_0_0 += A[(x + 0) * N + k_inner]
//                * B[(y + 0) * N + k_inner];
//            result_component_0_1 += A[(x + 0) * N + k_inner]
//                * B[(y + 1) * N + k_inner];
//            result_component_1_0 += A[(x + 1) * N + k_inner]
//                * B[(y + 0) * N + k_inner];
//            result_component_1_1 += A[(x + 1) * N + k_inner]
//                * B[(y + 1) * N + k_inner];
//
//            result_component_2_0 += A[(x + 2) * N + k_inner]
//                * B[(y + 0) * N + k_inner];
//            result_component_2_1 += A[(x + 2) * N + k_inner]
//                * B[(y + 1) * N + k_inner];
//            result_component_3_0 += A[(x + 3) * N + k_inner]
//                * B[(y + 0) * N + k_inner];
//            result_component_3_1 += A[(x + 3) * N + k_inner]
//                * B[(y + 1) * N + k_inner];
//
////            result_component_0_2 += A[(x + 0) * N + k_inner]
////                * B[(y + 2) * N + k_inner];
////            result_component_0_3 += A[(x + 0) * N + k_inner]
////                * B[(y + 3) * N + k_inner];
////            result_component_1_2 += A[(x + 1) * N + k_inner]
////                * B[(y + 2) * N + k_inner];
////            result_component_1_3 += A[(x + 1) * N + k_inner]
////                * B[(y + 3) * N + k_inner];
////
////            result_component_2_2 += A[(x + 2) * N + k_inner]
////                * B[(y + 2) * N + k_inner];
////            result_component_2_3 += A[(x + 2) * N + k_inner]
////                * B[(y + 3) * N + k_inner];
////            result_component_3_2 += A[(x + 3) * N + k_inner]
////                * B[(y + 2) * N + k_inner];
////            result_component_3_3 += A[(x + 3) * N + k_inner]
////                * B[(y + 3) * N + k_inner];
//          }
//          // assumes matrix was zero-initialized
//          C[(x + 0) * N + (y + 0)] += result_component_0_0;
//          C[(x + 0) * N + (y + 1)] += result_component_0_1;
//          C[(x + 1) * N + (y + 0)] += result_component_1_0;
//          C[(x + 1) * N + (y + 1)] += result_component_1_1;
//
//          C[(x + 2) * N + (y + 0)] += result_component_2_0;
//          C[(x + 2) * N + (y + 1)] += result_component_2_1;
//          C[(x + 3) * N + (y + 0)] += result_component_3_0;
//          C[(x + 3) * N + (y + 1)] += result_component_3_1;
//
////          C[(x + 0) * N + (y + 2)] += result_component_0_2;
////          C[(x + 0) * N + (y + 3)] += result_component_0_3;
////          C[(x + 1) * N + (y + 2)] += result_component_1_2;
////          C[(x + 1) * N + (y + 3)] += result_component_1_3;
////
////          C[(x + 2) * N + (y + 2)] += result_component_2_2;
////          C[(x + 2) * N + (y + 3)] += result_component_2_3;
////          C[(x + 3) * N + (y + 2)] += result_component_3_2;
////          C[(x + 3) * N + (y + 3)] += result_component_3_3;
////          }
//        }
//      }


#define X_REG 5
#define Y_REG 8
	  
	  using Vc::double_v;
	  for (size_t x = 0; x < N; x += X_REG) {
	    for (size_t y = 0; y < N; y += Y_REG) {

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
	      
	      double_v a_temp_1 = A_trans[0 * (N + PADDING) + (x + 0)];
	      double_v a_temp_2 = A_trans[0 * (N + PADDING) + (x + 1)];
	      double_v a_temp_3 = A_trans[0 * (N + PADDING) + (x + 2)];
	      double_v a_temp_4 = A_trans[0 * (N + PADDING) + (x + 3)];
	      // double_v a_temp_5 = A_trans[0 * (N + PADDING) + (x + 4)];

	      double_v b_temp_1 = double_v(&B_padded[0 * (N + PADDING) + y], Vc::Aligned);
	      double_v b_temp_2 = double_v(&B_padded[0 * (N + PADDING) + (y + 4)], Vc::Aligned);
	      
	      for (size_t k_inner = 0; k_inner < N; k_inner += 1) {

		// strangely no speedup?					      
		// // broadcast single value
		// double_v a_temp_1 = A[(x + 0) * N + k_inner];
		// double_v a_temp_2 = A[(x + 1) * N + k_inner];
		// double_v a_temp_3 = A[(x + 2) * N + k_inner];
		// double_v a_temp_4 = A[(x + 3) * N + k_inner];

		//TODO: assumes matrix is NOT transposed

		// double_v b_temp_1 = double_v(&B_padded[k_inner * (N + PADDING) + y], Vc::Aligned);
		// double_v b_temp_2 = double_v(&B_padded[k_inner * (N + PADDING) + (y + 4)], Vc::Aligned);

		// double_v a_temp_1 = A_trans[k_inner * (N + PADDING) + (x + 0)];
		// double_v a_temp_2 = A_trans[k_inner * (N + PADDING) + (x + 1)];
		// double_v a_temp_3 = A_trans[k_inner * (N + PADDING) + (x + 2)];
		// double_v a_temp_4 = A_trans[k_inner * (N + PADDING) + (x + 3)];
		// double_v a_temp_5 = A_trans[k_inner * (N + PADDING) + (x + 4)];
		
		acc_11 += a_temp_1 * b_temp_1;
		acc_12 += a_temp_1 * b_temp_2;
		
		acc_21 += a_temp_2 * b_temp_1;
		acc_22 += a_temp_2 * b_temp_2;

		a_temp_1 = A_trans[(k_inner + 1) * (N + PADDING) + (x + 0)];
		a_temp_2 = A_trans[(k_inner + 1) * (N + PADDING) + (x + 1)];
		
		acc_31 += a_temp_3 * b_temp_1;
		acc_41 += a_temp_4 * b_temp_1;
		acc_51 += a_temp_5 * b_temp_1;

		b_temp_1 = double_v(&B_padded[(k_inner + 1) * (N + PADDING) + y], Vc::Aligned);
		
		acc_32 += a_temp_3 * b_temp_2;		
		acc_42 += a_temp_4 * b_temp_2;	       
		acc_52 += a_temp_5 * b_temp_2;

		b_temp_2 = double_v(&B_padded[(k_inner + 1) * (N + PADDING) + (y + 4)], Vc::Aligned);

		a_temp_3 = A_trans[(k_inner + 1) * (N + PADDING) + (x + 2)];
		a_temp_4 = A_trans[(k_inner + 1) * (N + PADDING) + (x + 3)];
		a_temp_5 = A_trans[(k_inner + 1) * (N + PADDING) + (x + 4)];
	      }
	      acc_11.store(&C_padded[(x + 0) * (N + PADDING) + y]);
	      acc_21.store(&C_padded[(x + 1) * (N + PADDING) + y]);
	      acc_31.store(&C_padded[(x + 2) * (N + PADDING) + y]);
	      acc_41.store(&C_padded[(x + 3) * (N + PADDING) + y]);

	      acc_51.store(&C_padded[(x + 4) * (N + PADDING) + y]);

	      acc_12.store(&C_padded[(x + 0) * (N + PADDING) + (y + 4)]);
	      acc_22.store(&C_padded[(x + 1) * (N + PADDING) + (y + 4)]);
	      acc_32.store(&C_padded[(x + 2) * (N + PADDING) + (y + 4)]);
	      acc_42.store(&C_padded[(x + 3) * (N + PADDING) + (y + 4)]);

	      acc_52.store(&C_padded[(x + 4) * (N + PADDING) + (y + 4)]);
	    }
	    
	  }
	  
// #define X_REG 1
// #define Y_REG 4
// #define K_STEP 4
	  
// 	  using Vc::double_v;
// 	  for (size_t x = 0; x < N; x += X_REG) {
// 	    for (size_t y = 0; y < N; y += Y_REG) {

// 	      double_v acc[X_REG * Y_REG];
// 	      for (size_t i = 0; i < X_REG * Y_REG; i++) {x
// 		  acc[i] = 0.0;
// 	      }


// 	      for (size_t k_inner = 0; k_inner < N; k_inner += K_STEP) {

// 		double_v a_temp[X_REG];
// 		for (size_t i = 0; i < X_REG; i++) {
// 		  a_temp[i] = double_v(&A[(x + i) * N + k_inner]);
// 		}
// 		double_v b_temp[Y_REG];
// 		for (size_t j = 0; j < Y_REG; j++) {
// 		  b_temp[j] = double_v(&B[(y + j) * N + k_inner]);
// 		}
// 		for (size_t i = 0; i < X_REG; i++) {
// 		  for (size_t j = 0; j < Y_REG; j++) {
// 		    acc[i * Y_REG + j] += a_temp[i] * b_temp[j];
// 		  }
// 		}
// 	      }

// 	      for (size_t i = 0; i < X_REG; i++) {
// 		for (size_t j = 0; j < Y_REG; j++) {
// 		  C[(x + i) * N + (y + j)] = acc[i * Y_REG + j][0] + acc[i * Y_REG + j][1] + acc[i * Y_REG + j][2] + acc[i * Y_REG + j][3];
// 		}
// 	      }
// 	    }
// 	  }
        }


	for (size_t i = 0; i < N; i++) {
	  for (size_t j = 0; j < N; j++) {
	    C[i * N + j] = C_padded[i * (N + PADDING) + j];	    
	  }	
	}	
	
        return C;
    }
};

}
