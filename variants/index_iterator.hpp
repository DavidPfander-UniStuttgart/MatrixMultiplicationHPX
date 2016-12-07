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
#include <hpx/include/iostreams.hpp>

namespace index_iterator {

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

template<size_t dim, size_t cur_dim, typename T, typename F, typename ... Args>
typename std::enable_if<dim == 0, void>::type execute_looped(
    std::vector<T> &min, std::vector<T> &max, std::vector<T> &step, F f,
    Args ... args) {
  f(args...);
}

template<size_t dim, size_t cur_dim, typename T, typename F, typename ... Args>
typename std::enable_if<dim != 0, void>::type execute_looped(
    std::vector<T> &min, std::vector<T> &max, std::vector<T> &step, F f,
    Args ... args) {
  for (T cur = min[cur_dim]; cur < max[cur_dim]; cur += step[cur_dim]) {
    execute_looped<dim - 1, cur_dim + 1>(min, max, step, f, args..., cur);
  }
}

template<size_t dim, typename T, typename F>
void loop_nest(std::vector<T> &min, std::vector<T> &max,
    std::vector<T> &step, F f) {
  execute_looped<dim, 0>(min, max, step, f);
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

  // set steps for final (fast) iteration
  void set_final_steps(const std::vector<T> &steps) {
    std::get<0>(blocking_configuration[0]) = steps;
  }
private:

  size_t dim;

  std::vector<std::pair<std::vector<T>, std::vector<bool>>>blocking_configuration;
};

template<size_t dim, typename T, typename F>
void iterate_indices(blocking_pseudo_execution_policy<T> policy,
    std::vector<T> min, std::vector<T> max, F f) {
  if (min.size() != max.size() || min.size() != dim) {
    throw;
  }
//    size_t dim = min.size();
  auto pair = policy.pop();
  std::vector<T> &block = std::get<0>(pair);
  std::vector<bool> &parallel_dims = std::get<1>(pair);

  // hpx::cout << "inner min: ";
  // for (size_t i = 0; i < min.size(); i++) {
  //   if (i > 0) {
  //     hpx::cout << ", ";
  //   }
  //   hpx::cout << min[i];
  // }
  // hpx::cout << std::endl << hpx::flush;

  // hpx::cout << "inner max: ";
  // for (size_t i = 0; i < max.size(); i++) {
  //   if (i > 0) {
  //     hpx::cout << ", ";
  //   }
  //   hpx::cout << max[i];
  // }
  // hpx::cout << std::endl << hpx::flush;

  // hpx::cout << "block: ";
  // for (size_t i = 0; i < block.size(); i++) {
  //   if (i > 0) {
  //     hpx::cout << ", ";
  //   }
  //   hpx::cout << block[i];
  // }
  // hpx::cout << std::endl << "--------------------" << std::endl << hpx::flush;
  
  if (policy.is_last_blocking_step()) {
    // hpx::cout << "in final" << std::endl << hpx::flush;
//        dim_index_iterator<T> dim_iter(min, max, block);
//
//        size_t inner_index_count = 1;
//        for (size_t d = 0; d < dim; d++) {
//            inner_index_count *= (max[d] - min[d]) / block[d];
//        }
//
//        hpx::parallel::for_each_n(hpx::parallel::seq, dim_iter,
//                inner_index_count, f);
    loop_nest<dim>(min, max, block, f);
  } else {
    // hpx::cout << "not in final" << std::endl << hpx::flush;
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
        inner_index_count_reduced *= static_cast<size_t>(std::ceil(static_cast<double>(max[d] - min[d]) / static_cast<double>(block[d])));
      } else {
        inner_index_count_remain *= static_cast<size_t>(std::ceil(static_cast<double>(max[d] - min[d]) / static_cast<double>(block[d])));
      }
    }

    dim_index_iterator<T> dim_iter_reduced(min_reduced, max_reduced,
        block_reduced);

    // first process parallel dimensions
    hpx::parallel::for_each_n(hpx::parallel::par, dim_iter_reduced,
        inner_index_count_reduced,
        [parallel_dims_count, inner_index_count_remain, &policy, &map, &min, &max, &block, f](const std::vector<size_t> &partial_index) {
          std::vector<T> min_serial_fill(min);
          std::vector<T> max_serial_fill(max);
          map_dims(partial_index, map, min_serial_fill);
          map_dims(partial_index, map, max_serial_fill);

          // is an iterator only over the not yet processed dimensions
          dim_index_iterator<T> dim_iter_serial_fill(min_serial_fill, max_serial_fill, block);

          std::vector<T> recursive_min(dim);
          std::vector<T> recursive_max(dim);

	  // hpx::cout << "min_serial_fill: ";
	  // for (size_t i = 0; i < min_serial_fill.size(); i++) {
	  //   if (i > 0) {
	  //     hpx::cout << ", ";
	  //   }
	  //   hpx::cout << min_serial_fill[i];
	  // }
	  // hpx::cout << " ";
	  // hpx::cout << "max_serial_fill: ";
	  // for (size_t i = 0; i < max_serial_fill.size(); i++) {
	  //   if (i > 0) {
	  //     hpx::cout << ", ";
	  //   }
	  //   hpx::cout << max_serial_fill[i];
	  // }
	  // hpx::cout << std::endl << hpx::flush;
	  // hpx::cout << " ";
	  // hpx::cout << "block: ";
	  // for (size_t i = 0; i < block.size(); i++) {
	  //   if (i > 0) {
	  //     hpx::cout << ", ";
	  //   }
	  //   hpx::cout << block[i];
	  // }
	  // hpx::cout << std::endl << hpx::flush;
	  // hpx::cout << "inner_index_count_remain: " << inner_index_count_remain << std::endl << hpx::flush;

          hpx::parallel::for_each_n(hpx::parallel::seq, dim_iter_serial_fill,
              inner_index_count_remain,
              [&policy, &block, f, &recursive_min, &recursive_max](const std::vector<size_t> &cur_index) {
                // iterate within block
                for (size_t d = 0; d < dim; d++) {
                  recursive_min[d] = cur_index[d];
                }

                for (size_t d = 0; d < dim; d++) {
                  recursive_max[d] = cur_index[d] + block[d];
                }

//                                hpx::cout << "recursive_min: ";
//                                for (size_t i = 0; i < recursive_min.size(); i++) {
//                                    if (i > 0) {
//                                        hpx::cout << ", ";
//                                    }
//                                    hpx::cout << recursive_min[i];
//                                }
//                                hpx::cout << " ";
//                                hpx::cout << "recursive_max: ";
//                                for (size_t i = 0; i < recursive_max.size(); i++) {
//                                    if (i > 0) {
//                                        hpx::cout << ", ";
//                                    }
//                                    hpx::cout << recursive_max[i];
//                                }
//                                hpx::cout << std::endl << hpx::flush;

                // do recursive blocking
                iterate_indices<dim>(policy,
                    recursive_min, recursive_max, f);
              });

        });

  }
}

}
