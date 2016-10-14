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

namespace looped {

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

    dim_index_iterator(size_t dim, T max_index_1d) :
            dim(dim), cur_index(dim), max_index(dim) {
        // initialize index vector with default values
        std::fill(cur_index.begin(), cur_index.end(), T());
        std::fill(max_index.begin(), max_index.end(), max_index_1d);
    }

    dim_index_iterator(std::vector<T> cur_index,
            std::vector<T> max_index) :
            dim(cur_index.size()), cur_index(cur_index), max_index(max_index) {
        if (cur_index.size() != max_index.size()) {
            throw;
        }
    }
};

template<typename T, typename F>
void iterate_indices(std::vector<T> min, std::vector<T> max,
        std::vector<T> block, F f) {
    if (min.size() != max.size() || max.size() != block.size()) {
        throw;
    }
    size_t dim = min.size();
    dim_index_iterator<T> dim_iter(min, max);
    size_t index_count = std::inner_product(max.begin(), max.end(), min.begin(),
            1.0, std::multiplies<size_t>(), std::minus<size_t>());
    std::vector<bool> blocked_dim(dim);
    std::generate(blocked_dim.begin(), blocked_dim.end(), []() {
        static size_t i = 0;
        return block[i] != 1;
    });

    hpx::parallel::for_each_n(hpx::parallel::seq, dim_iter, index_count, f);
}

class matrix_multiply_looped {

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
    matrix_multiply_looped(size_t N, std::vector<double> &A,
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

        std::vector<size_t> min = { 0, 0, 0 };
        std::vector<size_t> max = { N, N, N };
        std::vector<size_t> block = { 1, 1, 1 };
//        std::vector<const hpx::parallel::parallel_execution_policy> policy = {
//                hpx::parallel::par, hpx::parallel::par, hpx::parallel::seq };

        iterate_indices<size_t>(min, max, block,
                [this, &C](const std::vector<size_t> &cur_index) {

                    size_t k_block = 1;
                    size_t x = cur_index[0];
                    size_t y = cur_index[1];
                    size_t k = cur_index[2];
                    hpx::cout << "x: " << x << " y: " << y << " k: " << k << std::endl << hpx::flush;
                    C[x * N + y] += A[x * N + k] * B[y * N + k];
//
//                    auto A_submatrix_begin = A.begin() + x * N + k_block;
//                    auto A_submatrix_end = A.begin() + x * N + k_block + block_input;
//                    // assumes transposed input matrix
//                    auto B_submatrix_begin = B.begin() + y * N + k_block;
//
//                    // do auto-vectorized small matrix dot product
//                    C[x * N + y] += std::inner_product(A_submatrix_begin,
//                            A_submatrix_end, B_submatrix_begin, 0.0);
                });

//        hpx::parallel::for_each_n(hpx::parallel::par,
//                dim_index_iterator<size_t>(2, result_blocks, 1),
//                result_blocks * result_blocks,
//                [this, &C](const std::vector<size_t>&cur_index) {
//
//                    size_t block_x = cur_index[0];
//                    size_t block_y = cur_index[1];
//                    // should the matrix be smaller than 128 (all bigger than 128 are also multiples of 128)
//                    size_t outer_x = block_x * block_result;
//                    size_t outer_y = block_y * block_result;
//
//                    // block the bands in the (big) input matrices, in this loop algorithm is within small block matrix
//                    for (size_t k_block = 0; k_block < N; k_block += block_input) {
//                        // process all (small) block result matrix components sequentially (to benefit from the cache within a single core)
//                        for (size_t x = outer_x; x < outer_x + block_result; x++) {
//                            for (size_t y = outer_y; y < outer_y + block_result; y++) {
////                                hpx::cout << "x: " << x << " y: " << y << std::endl << hpx::flush;
//                                auto A_submatrix_begin = A.begin() + x * N + k_block;
//                                auto A_submatrix_end = A.begin() + x * N + k_block + block_input;
//                                // assumes transposed input matrix
//                                auto B_submatrix_begin = B.begin() + y * N + k_block;
//
//                                // do auto-vectorized small matrix dot product
//                                C[x * N + y] += std::inner_product(A_submatrix_begin, A_submatrix_end, B_submatrix_begin, 0.0);
//                            }
//                        }
//                    }
//                });
        return C;
    }
};

}
