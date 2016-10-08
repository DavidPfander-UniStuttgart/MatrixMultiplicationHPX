/*
 * matrix_multiplication_component.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: pfandedd
 */

#include "matrix_multiply_recursive.hpp"

#include <hpx/include/lcos.hpp>
#include "matrix_multiply_util.hpp"
#include "matrix_multiply_multiplier.hpp"

HPX_REGISTER_COMPONENT(hpx::components::component<matrix_multiply_recursive>,
        matrix_multiply_recursive);

HPX_REGISTER_ACTION(matrix_multiply_recursive::distribute_recursively_action);

// HPX_REGISTER_ACTION(matrix_multiply_recursive::extract_submatrix_action);

// make async as well
void matrix_multiply_recursive::extract_submatrix(std::vector<double> &C,
        std::vector<double> C_small, size_t x, size_t y, size_t blockSize) {
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

std::vector<double> matrix_multiply_recursive::distribute_recursively(
        std::uint64_t x, std::uint64_t y, size_t blockSize) {

    if (verbose >= 1) {
        hpx::cout << hpx::find_here() << " work on x: " << x << ", y: " << y
                << " blockSize: " << blockSize << std::endl << hpx::flush;
    }
    if (blockSize <= small_block_size) {
        uint32_t comp_locality = hpx::get_locality_id();
        hpx::components::client<matrix_multiply_multiplier> multiplier;
        multiplier.connect_to("/multiplier#" + std::to_string(comp_locality));
        auto f = hpx::async<
                matrix_multiply_multiplier::calculate_submatrix_action>(
                multiplier.get_id(), x, y, blockSize);
        return f.get();
    } else {
        if (verbose >= 1) {
            hpx::cout << "handling large matrix, more work... (blocksize == "
                    << blockSize << ")" << std::endl << hpx::flush;
        }

        uint64_t submatrix_count = 4;
        uint64_t n_new = blockSize / 2;

        //TODO: why does this make a difference?
        // std::vector<hpx::id_type> node_ids = hpx::find_all_localities();

//    std::vector<hpx::components::client<matrix_multiply_recursive>> sub_recursives =
//      hpx::new_<hpx::components::client<matrix_multiply_recursive>[]>(
//								      // hpx::components::default_layout(node_ids),
//								      hpx::find_here(),
//								      submatrix_count, small_block_size, verbose).get();

        uint32_t comp_locality = hpx::get_locality_id();
        hpx::components::client<matrix_multiply_recursive> self;
        self.connect_to("/recursive#" + std::to_string(comp_locality));

        std::vector<std::tuple<size_t, size_t>> offsets = { { 0, 0 }, { 0
                + n_new, 0 }, { 0, 0 + n_new }, { 0 + n_new, 0 + n_new } };

        std::vector<double> C(blockSize * blockSize);

        std::vector<hpx::future<void>> g;
        for (size_t i = 0; i < submatrix_count; i++) {
            hpx::future<std::vector<double>> f = hpx::async<
                    matrix_multiply_recursive::distribute_recursively_action>(
                    self.get_id(), x + std::get<0>(offsets[i]),
                    y + std::get<1>(offsets[i]), n_new);
            g.push_back(
                    f.then(
                            hpx::util::unwrapped(
                                    [=,&C](std::vector<double> submatrix)
                                    {
                                        this->extract_submatrix(C, std::move(submatrix),
                                                std::get<0>(offsets[i]), std::get<1>(offsets[i]), n_new);
                                    })));
        }

        //TODO: try it with dataflow

        // wait for the matrix C to become ready
        hpx::wait_all(g);

        return C;
    }

    // hpx::cout << "---------------" << std::endl << "node mult:" << std::endl << hpx::flush;
    // hpx::cout << "blockSize: " << blockSize << " small_block_size: " << small_block_size << std::endl << hpx::flush;
    // 												     hpx::cout << "C.s: " << C.size() << std::endl << hpx::flush;
    // print_matrix(blockSize, C);
}
