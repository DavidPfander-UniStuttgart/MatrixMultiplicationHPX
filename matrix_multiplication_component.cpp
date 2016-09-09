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

		std::vector<hpx::id_type> node_ids = hpx::find_all_localities();

		std::vector<hpx::components::client<matrixMultiply_server>> sub_multipliers =
				hpx::new_<hpx::components::client<matrixMultiply_server>[]>(
						hpx::components::default_layout(node_ids), submatrix_count, N, A, B).get();

		std::vector<std::tuple<size_t, size_t>> offsets = { { 0, 0 }, { 0
				+ n_new, 0 }, { 0, 0 + n_new }, { 0 + n_new, 0 + n_new } };

		std::vector<hpx::future<void>> g;
		for (size_t i = 0; i < submatrix_count; i++) {
			hpx::future<std::vector<double>> f = hpx::async<matrixMultiply_server::matrixMultiply_action>(sub_multipliers[i].get_id(), x + std::get<0>(offsets[i]),
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

//TODO: try it with dataflow

// wait for the matrix C to become ready
		hpx::wait_all(g);
	}
	return C;
}
