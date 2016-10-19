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

namespace par {

template<typename T>
void map_dims(const std::vector<T> &sub_index, const std::vector<T> &map,
        std::vector<T> &index) {
    if (sub_index.size() != map.size()) {
        throw;
    }
    for (size_t i = 0; i < sub_index.size(); i++) {
        index.at(map[i]) = sub_index.at(i);
    }
}

template<typename T>
class dim_index_iterator: public hpx::util::iterator_facade<
        dim_index_iterator<T>, std::vector<T>, std::forward_iterator_tag,
        const std::vector<T>&> {
private:

//    typedef hpx::util::iterator_facade<dim_index_iterator<T>, std::vector<T>,
//            std::forward_iterator_tag, const std::vector<T>&> base_type;

    size_t dim;
    std::vector<T> cur_index;
    std::vector<T> &min_index;
    std::vector<T> &max_index;
    std::vector<T> &step;

    friend class hpx::util::iterator_core_access;

// end check is definitely missing

    void increment() {
        for (size_t d = 0; d < dim; d++) {
//            hpx::cout << "d: " << d << std::endl << hpx::flush;
            if (cur_index[d] + step[d] < max_index[d]) {
//                hpx::cout << "found, incrementing d = " << d << " val: " << cur_index[d] << " (max_index[d] = " << max_index[d] << ")" << std::endl << hpx::flush;
                cur_index[d] += step[d];

                // reset lower dimensions
                for (size_t i = 0; i < d; i++) {
//                    hpx::cout << "resetting d = " << i << std::endl << hpx::flush;
                    cur_index[i] = min_index[i];
                }
                return;
            }
        }
        // after last element in iteration -> happens at the end
        //TODO: add proper end() treatment
    }

    bool equal(dim_index_iterator const& other) const {
        return std::equal(cur_index.begin(), cur_index.end(),
                other.cur_index.begin());
    }

    const std::vector<T> &dereference() const {
        return cur_index;
    }

public:
//    dim_index_iterator() :
//            dim(0), cur_index(0),  {
//    }

//    dim_index_iterator(size_t dim, T max_index_1d) :
//            dim(dim), cur_index(dim), min_index(dim), max_index(dim), step(dim) {
//        // initialize index vector with default values
//        std::fill(cur_index.begin(), cur_index.end(), T());
//        std::fill(min_index.begin(), min_index.end(), T());
//        std::fill(max_index.begin(), max_index.end(), max_index_1d);
//        std::fill(step.begin(), step.end(), static_cast<T>(1));
//    }

    dim_index_iterator(std::vector<T> &min_index, std::vector<T> &max_index,
            std::vector<T> &step) :
            dim(min_index.size()), cur_index(min_index), min_index(min_index), max_index(
                    max_index), step(step) {
        if (cur_index.size() != max_index.size()
                || max_index.size() != step.size()) {
            throw;
        }
    }

    dim_index_iterator &operator=(dim_index_iterator const &other) {
        if (this != &other) {
            dim = other.dim;
            cur_index = other.cur_index;
            min_index = other.min_index;
            max_index = other.max_index;
            step = other.step;
        }
        return *this;
    }
};

template<typename T>
class blocking_pseudo_execution_policy {
public:
    blocking_pseudo_execution_policy(size_t dim) :
            dim(dim) {
        this->add_blocking(std::vector<T>(dim, static_cast<T>(1)),
                std::vector<bool>(dim, false));
    }

    blocking_pseudo_execution_policy &add_blocking(const std::vector<T> &block,
            const std::vector<bool> &parallel_dims) {
        if (block.size() != dim || parallel_dims.size() != dim) {
            throw;
        }
        blocking_configuration.push_back(std::make_pair(block, parallel_dims));
        return *this;
    }

    std::pair<std::vector<T>, std::vector<bool>> pop() {
        auto last = blocking_configuration.back();
        blocking_configuration.pop_back();
        return last;
    }

    bool is_last_blocking_step() {
        return blocking_configuration.size() == 0;
    }
private:

    size_t dim;

    std::vector<std::pair<std::vector<T>, std::vector<bool>>>blocking_configuration;
};

template<typename T, typename F>
void iterate_indices(blocking_pseudo_execution_policy<T> policy,
        std::vector<T> min, std::vector<T> max, F f) {
    if (min.size() != max.size()) {
        throw;
    }
    size_t dim = min.size();
    auto pair = policy.pop();
    std::vector<T> &block = std::get<0>(pair);
    std::vector<bool> &parallel_dims = std::get<1>(pair);

    if (policy.is_last_blocking_step()) {
        dim_index_iterator<T> dim_iter(min, max, block);

        size_t inner_index_count = 1;
        for (size_t d = 0; d < dim; d++) {
            inner_index_count *= (max[d] - min[d]) / block[d];
        }

        hpx::parallel::for_each_n(hpx::parallel::seq, dim_iter,
                inner_index_count, f);
    } else {
        size_t parallel_dims_count = std::count(parallel_dims.begin(),
                parallel_dims.end(), true);

        std::vector<T> map;
        map.reserve(parallel_dims_count);
        for (size_t d = 0; d < dim; d++) {
            if (parallel_dims.at(d)) {
                map.push_back(d);
            }
        }

        std::vector<T> min_reduced;
        min_reduced.reserve(parallel_dims_count);
        for (size_t d = 0; d < dim; d++) {
            if (parallel_dims.at(d)) {
                min_reduced.push_back(min[d]);
            }
        }

        std::vector<T> max_reduced;
        max_reduced.reserve(parallel_dims_count);
        for (size_t d = 0; d < dim; d++) {
            if (parallel_dims.at(d)) {
                max_reduced.push_back(max[d]);
            }
        }

        std::vector<T> block_reduced;
        block_reduced.reserve(parallel_dims_count);
        for (size_t d = 0; d < dim; d++) {
            if (parallel_dims.at(d)) {
                block_reduced.push_back(block[d]);
            }
        }

        size_t inner_index_count_reduced = 1;
        size_t inner_index_count_remain = 1;
        for (size_t d = 0; d < dim; d++) {
            if (parallel_dims[d]) {
                inner_index_count_reduced *= (max[d] - min[d]) / block[d];
            } else {
                inner_index_count_remain *= (max[d] - min[d]) / block[d];

            }
        }

        dim_index_iterator<T> dim_iter_reduced(min_reduced, max_reduced,
                block_reduced);

        // first process parallel dimensions
        hpx::parallel::for_each_n(hpx::parallel::par, dim_iter_reduced,
                inner_index_count_reduced,
                [dim, parallel_dims_count, inner_index_count_remain, &policy, &map, &min, &max, &block, f](const std::vector<size_t> &partial_index) {

                    std::vector<T> min_serial_fill(min);
                    std::vector<T> max_serial_fill(max);
                    map_dims(partial_index, map, min_serial_fill);
                    map_dims(partial_index, map, max_serial_fill);

                    // is an iterator only over the not yet processed dimensions
                    dim_index_iterator<T> dim_iter_serial_fill(min_serial_fill, max_serial_fill, block);

                    std::vector<T> recursive_min(dim);
                    std::vector<T> recursive_max(dim);

                    hpx::parallel::for_each_n(hpx::parallel::seq, dim_iter_serial_fill,
                            inner_index_count_remain,
                            [dim, &policy, &block, f, &recursive_min, &recursive_max](const std::vector<size_t> &cur_index) {
                                // iterate within block

                                for (size_t d = 0; d < dim; d++) {
                                    recursive_min[d] = cur_index[d];
                                }

                                for (size_t d = 0; d < dim; d++) {
                                    recursive_max[d] = cur_index[d] + block[d];
                                }

                                // do recursive blocking
                                iterate_indices(policy,
                                        recursive_min, recursive_max, f);
                            });

                });

    }
}

class matrix_multiply_par {

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
    matrix_multiply_par(size_t N, std::vector<double> &A,
            std::vector<double> &B, bool transposed, uint64_t block_result,
            uint64_t block_input, uint64_t repetitions, uint64_t verbose) :
            N(N), A(A), B(B), transposed(transposed), block_result(
                    block_result), block_input(block_input), repetitions(
                    repetitions), verbose(verbose) {

    }

    std::vector<double> matrix_multiply() {
        std::vector<double> C(N * N);
        std::fill(C.begin(), C.end(), 0.0);

        std::vector<size_t> min = { 0, 0, 0 };
        std::vector<size_t> max = { N, N, N };
        std::vector<size_t> block = { block_result, block_result, block_input };
        std::vector<bool> parallel_dims = { true, true, false };

//        hpx::parallel::par;
//        const hpx::parallel::parallel_execution_policy &t = hpx::parallel::seq;

//        std::vector<
//                std::reference_wrapper<
//                        const hpx::parallel::parallel_execution_policy>>
//        execution_policy =
//        {   hpx::parallel::par, hpx::parallel::par, hpx::parallel::seq};

        blocking_pseudo_execution_policy<size_t> policy(3);
        policy.add_blocking(block, parallel_dims);

//        compute_kernel_struct compute_kernel(A, B, C, N);
//
//        action_wrapper<size_t, compute_kernel_struct> wrap;
        iterate_indices(policy, min, max,
                [this, &C](const std::vector<size_t> &cur_index) {

                    size_t x = cur_index[0];
                    size_t y = cur_index[1];
                    size_t k = cur_index[2];
//                    hpx::cout << "x: " << x << " y: " << y << " k: " << k << std::endl << hpx::flush;
                    C[x * N + y] += A[x * N + k] * B[y * N + k];
                });
        return C;
    }
};

}
