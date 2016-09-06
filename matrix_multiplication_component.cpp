/*
 * matrix_multiplication_component.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: pfandedd
 */

#include "matrix_multiplication_component.hpp"

// move to cpp
HPX_REGISTER_COMPONENT(hpx::components::component<matrixMultiply_server>,
		matrixMultiply_server);

// move to cpp
HPX_REGISTER_ACTION(matrixMultiply_server::matrixMultiply_action);

std::vector<double> matrixMultiply_server::matrixMultiply(std::uint64_t x,
		std::uint64_t y, size_t blockSize) {
	std::vector<double> C(blockSize * blockSize);
	if (blockSize <= small_block_size) {
		kernel::matrix_multiply_kernel(A, B, C, N, x, y, blockSize);
	} else {
		if (verbose >= 1) {
			hpx::cout << "handling large matrix, more work... (blocksize == "
					<< blockSize << ")" << std::endl << hpx::flush;
		}
		// We restrict ourselves to execute the matrixMultiply function locally.
//		hpx::naming::id_type const locality_id = hpx::find_here();
		uint64_t n_new = blockSize / 2;

//		hpx::future<hpx::id_type> id_future = hpx::new_<matrixMultiply_server>(hpx::find_here(), N, A, B);
//		hpx::components::client<matrixMultiply_server> cl = id_future.get();
//		hpx::components::client<matrixMultiply_server> multiplier2 = hpx::new_<matrixMultiply_server>(hpx::find_here(), N, A, B);
//		hpx::components::client<matrixMultiply_server> multiplier3 = hpx::new_<matrixMultiply_server>(hpx::find_here(), N, A, B);
//		hpx::components::client<matrixMultiply_server> multiplier4 = hpx::new_<matrixMultiply_server>(hpx::find_here(), N, A, B);
//

		matrixMultiply_client multiplier1 = matrixMultiply_client::create(
				hpx::find_here(), N, A, B);
		hpx::future<std::vector<double>> f1 = multiplier1.matrixMultiplyClient(
				x, y, n_new);
		matrixMultiply_client multiplier2 = matrixMultiply_client::create(
				hpx::find_here(), N, A, B);
		hpx::future<std::vector<double>> f2 = multiplier2.matrixMultiplyClient(
				x + n_new, y, n_new);
		matrixMultiply_client multiplier3 = matrixMultiply_client::create(
				hpx::find_here(), N, A, B);
		hpx::future<std::vector<double>> f3 = multiplier3.matrixMultiplyClient(
				x, y + n_new, n_new);
		matrixMultiply_client multiplier4 = matrixMultiply_client::create(
				hpx::find_here(), N, A, B);
		hpx::future<std::vector<double>> f4 = multiplier4.matrixMultiplyClient(
				x + n_new, y + n_new, n_new);

//		// split the current submatrix into four further submatrices
//		matrixMultiply_action mat;
//		std::vector<double> f1 = hpx::async(mat, locality_id, x, y, n_new);
//		std::vector<double> f2 = hpx::async(mat, locality_id, x + n_new, y,
//				n_new);
//		std::vector<double> f3 = hpx::async(mat, locality_id, x, y + n_new,
//				n_new);
//		std::vector<double> f4 = hpx::async(mat, locality_id, x + n_new,
//				y + n_new, n_new);
//
		std::vector<double> C_small;
		C_small = f1.get();
		kernel::extract_submatrix(C, C_small, 0, 0, n_new);
		C_small = f2.get();
		kernel::extract_submatrix(C, C_small, 0 + n_new, 0, n_new);
		C_small = f3.get();
		kernel::extract_submatrix(C, C_small, 0, 0 + n_new, n_new);
		C_small = f4.get();
		kernel::extract_submatrix(C, C_small, 0 + n_new, 0 + n_new, n_new);
	}
	return C;
}
