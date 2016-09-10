/*
 * matrix_multiplication_component.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: pfandedd
 */

#include <hpx/include/lcos.hpp>
#include "matrix_multiply_node.hpp"

HPX_REGISTER_COMPONENT(hpx::components::component<matrix_multiply_node>,
		matrix_multiply_node);

HPX_REGISTER_ACTION(matrix_multiply_node::matrix_multiply_action);

HPX_REGISTER_ACTION(matrix_multiply_node::extract_submatrix_action);

// make async as well
void matrix_multiply_node::extract_submatrix(std::vector<double> C_small,
		size_t x, size_t y, size_t blockSize) {
	try {
		size_t blocksize_last = 2 * blockSize;
		for (uint64_t i = 0; i < blockSize; i++) {
			for (uint64_t j = 0; j < blockSize; j++) {
//				std::cout << "x: " << x << " y: " << y << " i: " << i << " j: "
//						<< j << " blockSize: " << blockSize << " N: " << N
//						<< std::endl;
				C.at((x + i) * blocksize_last + (y + j)) = C_small.at(
						i * blockSize + j);
			}
		}
	} catch (const std::out_of_range &oor) {
		std::cout << "in extract_submatrix: " << oor.what() << std::endl;
	}
}

std::vector<double> matrix_multiply_node::matrix_multiply(std::uint64_t x,
		std::uint64_t y, size_t blockSize) {
	C.resize(blockSize * blockSize);
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

		std::vector<hpx::components::client<matrix_multiply_node>> sub_multipliers =
				hpx::new_<hpx::components::client<matrix_multiply_node>[]>(
						hpx::components::default_layout(node_ids),
						submatrix_count, N, A, B).get();

		std::vector<std::tuple<size_t, size_t>> offsets = { { 0, 0 }, { 0
				+ n_new, 0 }, { 0, 0 + n_new }, { 0 + n_new, 0 + n_new } };

		std::vector<hpx::future<void>> g;
		for (size_t i = 0; i < submatrix_count; i++) {
			//TODO: how do I do this with async_continue, make extract_submatrix asynchronous
			hpx::future<std::vector<double>> f = hpx::async<
					matrix_multiply_node::matrix_multiply_action>(
					sub_multipliers[i].get_id(), x + std::get<0>(offsets[i]),
					y + std::get<1>(offsets[i]), n_new);
			g.push_back(
					f.then(
							hpx::util::unwrapped(
									[=](std::vector<double> submatrix)
									{
										this->extract_submatrix(std::move(submatrix), std::get<0>(offsets[i]), std::get<1>(offsets[i]), n_new);
									})));
		}

//TODO: try it with dataflow

// wait for the matrix C to become ready
		hpx::wait_all(g);
	}
	return C;
}
