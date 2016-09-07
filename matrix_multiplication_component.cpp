/*
 * matrix_multiplication_component.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: pfandedd
 */

#include "matrix_multiplication_component.hpp"
#include <hpx/include/lcos.hpp>

HPX_REGISTER_COMPONENT(hpx::components::component<matrixMultiply_server>,
		matrixMultiply_server);

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

		uint64_t submatrix_count = 4;
		uint64_t n_new = blockSize / 2;

//		hpx::future<hpx::id_type> id_future = hpx::new_<matrixMultiply_server>(hpx::find_here(), N, A, B);
//		hpx::components::client<matrixMultiply_server> cl = id_future.get();

		//TODO: major problem: how to use the same policy information on every node?
		std::vector<hpx::id_type> node_ids = hpx::find_all_localities();
		hpx::components::default_distribution_policy policy;
		// TODO: is that required? -> constructor is proteced
		policy(node_ids);

		std::vector<matrixMultiply_client> sub_multipliers =
				hpx::new_<matrixMultiply_client[]>(policy, submatrix_count, N,
						A, B).get();

		// offsets for submatrices
//		std::vector<std::tuple<size_t, size_t>> offsets = { { x, y }, { x
//				+ n_new, y }, { x, y + n_new }, { x + n_new, y + n_new } };

		std::vector<std::tuple<size_t, size_t>> offsets = { { 0, 0 }, { 0
				+ n_new, 0 }, { 0, 0 + n_new }, { 0 + n_new, 0 + n_new } };

		std::vector<hpx::future<void>> g;
		for (size_t i = 0; i < submatrix_count; i++) {
			hpx::future<std::vector<double>> f =
					sub_multipliers[i].matrixMultiplyClient(
							x + std::get<0>(offsets[i]),
							y + std::get<1>(offsets[i]), n_new);

			g.push_back(
					f.then(
							hpx::util::unwrapped(
									[=,&C](std::vector<double> submatrix)
									{
										kernel::extract_submatrix(C, submatrix, std::get<0>(offsets[i]), std::get<1>(offsets[i]), n_new);
									})));
		}

//		for (size_t i = 0; i < submatrix_count; i++) {
//			std::vector<double> C_small = result_proxies[i].get();
//			kernel::extract_submatrix(C, C_small, std::get<0>(offsets_restore[i]), std::get<1>(offsets_restore[i]), n_new);
//		}

// wait for the matrix C to become ready
		hpx::wait_all(g);
	}
	return C;
}
