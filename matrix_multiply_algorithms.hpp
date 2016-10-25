/*
 * matrix_multiply_algorithms.hpp
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

namespace algorithms {

template<typename T>
class index_iterator: public boost::iterator_facade<index_iterator<T>,
        std::tuple<size_t, size_t, std::reference_wrapper<T>>,
        std::forward_iterator_tag,
        std::tuple<size_t, size_t, std::reference_wrapper<T>>>{
private:
size_t stride;
typename std::vector<T>::iterator wrapped_iterator;
size_t x;
size_t y;

friend class boost::iterator_core_access;

void increment() {
    wrapped_iterator++;
    if (y == stride - 1) {
        x += 1;
        y = 0;
    } else {
        y += 1;
    }
}

bool equal(index_iterator const& other) const {
    return this->wrapped_iterator == other.wrapped_iterator;
}

std::tuple<size_t, size_t, std::reference_wrapper<T>> dereference() const {
    return std::make_tuple(x, y, std::ref(*wrapped_iterator));
}

public:
index_iterator() :
stride(0), x(0), y(0) {
}

index_iterator(std::vector<T> &container, size_t stride) :
stride(stride), wrapped_iterator(container.begin()), x(0), y(0) {

}
};

template<typename T>
class dim_index_iterator: public hpx::util::iterator_facade<
        dim_index_iterator<T>, std::vector<T>, std::forward_iterator_tag,
        const std::vector<T>&> { //, const std::vector<T>&
private:

//    typedef hpx::util::iterator_facade<dim_index_iterator<T>, std::vector<T>,
//            std::forward_iterator_tag, const std::vector<T>&> base_type;

    size_t dim;
    std::vector<T> cur_index;
    std::vector<T> max_index;

    friend class hpx::util::iterator_core_access;

// end check is definitely missing

    void increment() {
        for (size_t d = 0; d < dim; d++) {
//            hpx::cout << "d: " << d << std::endl << hpx::flush;
            if (cur_index[d] + static_cast<T>(1) < max_index[d]) {
//                hpx::cout << "found, incrementing d = " << d << " val: " << cur_index[d] << " (max_index[d] = " << max_index[d] << ")" << std::endl << hpx::flush;
                cur_index[d]++;

                // reset lower dimensions
                for (size_t i = 0; i < d; i++) {
//                    hpx::cout << "resetting d = " << i << std::endl << hpx::flush;
                    cur_index[i] = T();
                }
                return;
            }
        }
        // after last element in iteration
//        throw;
    }

    bool equal(dim_index_iterator const& other) const {
        return std::equal(cur_index.begin(), cur_index.end(),
                other.cur_index.begin());
//    return this->cur_index == other.cur_index;
    }

//    typename base_type::reference dereference() const {
//        return cur_index;
//    }

    const std::vector<T> &dereference() const {
        return cur_index;
    }

public:
    dim_index_iterator() :
            dim(0), cur_index(0) {
    }

    dim_index_iterator(size_t dim, T max_index_1d, T stride) :
            dim(dim), cur_index(dim), max_index(dim) {
        // initialize index vector with default values
        std::fill(cur_index.begin(), cur_index.end(), T());
        std::fill(max_index.begin(), max_index.end(), max_index_1d);
    }
};

class matrix_multiply_algorithms {

private:
    size_t N;
    std::vector<double> &A;
    std::vector<double> &B;
    bool transposed;
    uint64_t block_input;
    size_t block_result;

    uint64_t repetitions;
    uint64_t verbose;
public:
    matrix_multiply_algorithms(size_t N, std::vector<double> &A,
            std::vector<double> &B, bool transposed, uint64_t block_input,
            size_t small_block_size, uint64_t repetitions, uint64_t verbose) :
            N(N), A(A), B(B), transposed(transposed), block_input(block_input), block_result(
                    small_block_size), repetitions(repetitions), verbose(
                    verbose) {

    }

    std::vector<double> matrix_multiply() {
        std::vector<double> C(N * N);
        std::fill(C.begin(), C.end(), 0.0);

        // correct number of blocks if N % result_blocks == 0
        size_t result_blocks = (N / block_result);

        // round up if N % result_blocks != 0
        if (N % block_result != 0) {
            result_blocks += 1;
        }

        if (verbose >= 1) {
            hpx::cout << "result_blocks: " << result_blocks << std::endl
                    << hpx::flush;
        }

//        dim_index_iterator<size_t> test(2, result_blocks, 1);
//        const std::vector<size_t> &ele = *test;
////        test.operator
//        hpx::parallel::for_each_n(hpx::parallel::seq,
//                test,
//                result_blocks * result_blocks,
//                [](const std::vector<size_t> &t) {});

        // get 2D indices (block indices)
//        hpx::parallel::for_loop(hpx::parallel::par, 0, result_blocks,
//                [this, result_blocks, &C](size_t block_x) {
//                    hpx::parallel::for_loop(hpx::parallel::par, 0, result_blocks, [this, block_x, &C](size_t block_y) {
        hpx::parallel::for_each_n(hpx::parallel::par,
                dim_index_iterator<size_t>(2, result_blocks, 1),
                result_blocks * result_blocks,
                [this, &C](const std::vector<size_t>&cur_index) {

                    size_t block_x = cur_index[0];
                    size_t block_y = cur_index[1];
//                    hpx::cout << "block_x: " << block_x << " block_y: " << block_y << std::endl << hpx::flush;
                    // should the matrix be smaller than 128 (all bigger than 128 are also multiples of 128)
                    size_t outer_x = block_x * block_result;
                    size_t outer_y = block_y * block_result;

                    // use small intermediate matrices to reduce conflict misses
                    std::vector<double> C_inner(block_result * block_result);
                    std::fill(C_inner.begin(), C_inner.end(), 0.0);

                    // use small intermediate matrices to reduce conflict misses
                    std::vector<double> A_small(block_input * block_result);

                    // use small intermediate matrices to reduce conflict misses
                    std::vector<double> B_small(block_input * block_result);

                    // block the bands in the (big) input matrices, in this loop algorithm is within small block matrix
                    for (size_t k_block = 0; k_block < N; k_block += block_input) {
                        // should the matrix be smaller than 128 (all bigger than 128 are also multiples of 128)
//                        size_t k_input_end = std::min(k_block + block_input, N);

                        // move input matrix into cache (and suppress conflict misses)
                        for (size_t x = 0; x < block_result; x++) {
                            for (size_t k = 0; k < block_input; k++) {
                                A_small[x * block_input + k] = A[(outer_x + x) * N + k_block + k];
                            }
                        }

                        // move input matrix into cache (and suppress conflict misses)
                        for (size_t y = 0; y < block_result; y++) {
                            for (size_t k = 0; k < block_input; k++) {
                                B_small[y * block_input + k] = B[(outer_y + y) * N + k_block + k];
                            }
                        }

                        // process all (small) block result matrix components sequentially (to benefit from the cache within a single core)
                        hpx::parallel::for_each_n(hpx::parallel::seq,
                                index_iterator<double>(C_inner, block_result), block_result * block_result,
                                [this, &A_small, &B_small](std::tuple<size_t, size_t, std::reference_wrapper<double>> t) {
                                    size_t x = std::get<0>(t);
                                    size_t y = std::get<1>(t);
                                    double &value = std::get<2>(t);

                                    auto A_submatrix_begin = A_small.begin() + x * block_input;
                                    auto A_submatrix_end = A_small.begin() + (x + 1) * block_input;
                                    // assumes transposed input matrix
                                    auto B_submatrix_begin = B_small.begin() + y * block_input;

                                    // do auto-vectorized small matrix dot product
                                    value += std::inner_product(A_submatrix_begin, A_submatrix_end, B_submatrix_begin, 0.0);
                                });
                    }

                    // write (small) block result matrix back into (big) overall result matrix
                    for (size_t x = 0; x < block_result; x++) {
                        for (size_t y = 0; y < block_result; y++) {
                            C[(outer_x + x) * N + (outer_y + y)] = C_inner[x * block_result + y];
                        }
                    }
//                            });
                });
        return C;
    }
};

}
