/*
 * matrix_multiplication_component.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: pfandedd
 */

#pragma once

#include <cinttypes>
#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>

#include "matrix_multiply_kernel.hpp"

struct matrix_multiply_recursive: hpx::components::component_base<
        matrix_multiply_recursive> {

    size_t block_result;
    uint64_t verbose;

    // TODO: why does this get called?
    matrix_multiply_recursive() :
        block_result(1), verbose(0) {
    }

    matrix_multiply_recursive(size_t block_result, uint64_t verbose) :
        block_result(block_result), verbose(verbose) {
    }

    ~matrix_multiply_recursive() {
    }

    std::vector<double> distribute_recursively(std::uint64_t x, std::uint64_t y,
            size_t blockSize);

    void extract_submatrix(std::vector<double> &C, std::vector<double> C_small,
            size_t x, size_t y, size_t blockSize);

    HPX_DEFINE_COMPONENT_ACTION(matrix_multiply_recursive, distribute_recursively,
            distribute_recursively_action);

    // HPX_DEFINE_COMPONENT_ACTION(matrix_multiply_recursive, extract_submatrix,
    // 			      extract_submatrix_action);

};

HPX_REGISTER_ACTION_DECLARATION(
        matrix_multiply_recursive::distribute_recursively_action);
