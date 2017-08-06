#include "static_improved.hpp"

#include <algorithm>

#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/lcos.hpp>

#include "components/multiplier.hpp"
#include "components/recursive.hpp"

namespace multiply_components {

void static_improved::print_schedule(
    std::vector<std::vector<matrix_multiply_work>> &all_work) {
  std::cout << "-----work-distribution:-----" << std::endl;
  for (size_t i = 0; i < all_work.size(); i++) {
    std::vector<matrix_multiply_work> &locality_work = all_work[i];
    std::cout << "locality: " << i << " = ";
    for (size_t j = 0; j < locality_work.size(); j++) {
      if (j > 0) {
        std::cout << ", ";
      }
      std::cout << "[" << locality_work[j].N << "]";
    }
    std::cout << std::endl;
  }
}

void static_improved::insert_submatrix(const std::vector<double> &submatrix,
                                       const matrix_multiply_work &w) {
  for (size_t i = 0; i < w.N; i++) {
    for (size_t j = 0; j < w.N; j++) {
      // std::cout << (w.x + i) << ", " << (w.y + j) << " val: " << submatrix[i
      // * w.N + j] << std::endl;
      C[(w.x + i) * N + (w.y + j)] = submatrix[i * w.N + j];
    }
  }
}

bool static_improved::fulfills_constraints(std::vector<uint64_t> &total_work) {
  uint64_t min_work = total_work[0];
  uint64_t max_work = total_work[0];

  for (size_t i = 1; i < total_work.size(); i++) {
    if (total_work[i] < min_work) {
      min_work = total_work[i];
    }
    if (total_work[i] > max_work) {
      max_work = total_work[i];
    }
  }

  // std::cout << "+++ work_difference: " << (max_work - min_work) << std::endl;

  if (max_work - min_work < max_work_difference) {
    return true;
  }
  if (min_work > 0 && max_work > 0) {
    double relative_work_difference =
        (static_cast<double>(max_work - min_work) /
         static_cast<double>(min_work));
    // std::cout << "+++ relative_work_difference: "
    // 	  << relative_work_difference << " bound: " <<
    // max_relative_work_difference << std::endl;
    if (relative_work_difference < max_relative_work_difference) {
      return true;
    }
  }
  return false;
}

std::vector<std::vector<matrix_multiply_work>>
static_improved::create_schedule(size_t num_localities) {
  // one slot per locality, stores the work to do
  std::vector<std::vector<matrix_multiply_work>> all_work(num_localities);

  // map: locality -> assigned components
  std::vector<uint64_t> total_work(num_localities);

  // initially all work is placed on the first locality
  all_work[0].emplace_back(0, 0, N);
  total_work[0] = N * N;

  while (!this->fulfills_constraints(total_work)) {

    // give some work from the max element to the min element
    uint64_t max_work = total_work[0];
    // is a vector, as there could be multiple indices with max work value
    // test every of them, as some might be invalid due to no existing work
    // packages to distribute
    std::vector<size_t> max_indices = {0};
    uint64_t min_work = total_work[0];
    size_t min_index = 0;

    for (size_t i = 1; i < num_localities; i++) {
      if (total_work[i] > max_work) {
        max_work = total_work[i];
        max_indices.clear();
        max_indices.push_back(i);
      } else if (total_work[i] == max_work) {
        max_indices.push_back(i);
      }

      if (total_work[i] < min_work) {
        min_work = total_work[i];
        min_index = i;
      }
    }

    // std::cout << "max_work: " << max_work << std::endl;
    // std::cout << "min_work: " << min_work << std::endl;

    bool found = false;
    size_t valid_max_index = 0;
    size_t valid_work_to_balance_index = 0;

    // std::cout << "max_indices: ";
    // for (size_t i = 0; i < max_indices.size(); i++) {
    //   if (i > 0) {
    //     std::cout << ", ";
    //   }
    //   std::cout << max_indices[i];
    // }
    // std::cout << std::endl;

    for (size_t max_index : max_indices) {

      // will always succeed, do not have to skip first iteration
      size_t work_to_balance_index = all_work[max_index].size();
      uint64_t smallest_work_difference = max_work - min_work;
      for (size_t j = 0; j < all_work[max_index].size(); j++) {
        matrix_multiply_work &w = all_work[max_index][j];
        if (w.N <= this->min_work_size) {
          continue;
        }
        // split work package between both locations
        uint64_t split_size = (w.N * w.N) / 2;
        // std::cout << "smallest_work_difference: " << smallest_work_difference
        // << std::endl;
        // std::cout << "split_size: " << split_size << std::endl;
        int64_t work_difference =
            std::abs(static_cast<int64_t>(max_work - split_size) -
                     static_cast<int64_t>(min_work + split_size));
        // std::cout << "work_difference: " << work_difference << std::endl;
        if (static_cast<uint64_t>(work_difference) <=
            smallest_work_difference) {
          smallest_work_difference = work_difference;
          work_to_balance_index = j;
        }
      }

      // work found -> apply
      if (work_to_balance_index != all_work[max_index].size()) {
        found = true;
        valid_max_index = max_index;
        valid_work_to_balance_index = work_to_balance_index;
        break;
      }
    }

    if (!found) {
      if (verbose >= 1) {
        std::cout << "aborted due to no valid work package to distribute found"
                  << std::endl;
      }
      break;
    }

    matrix_multiply_work w =
        all_work[valid_max_index][valid_work_to_balance_index];
    all_work[valid_max_index].erase(all_work[valid_max_index].begin() +
                                    valid_work_to_balance_index);

    size_t n_new = w.N / 2;
    // std::vector<std::tuple<size_t, size_t>> offsets = {
    //     {0, 0}, {0 + n_new, 0}, {0, 0 + n_new}, {0 + n_new, 0 + n_new}};
    std::vector<std::tuple<size_t, size_t>> offsets;
    offsets.emplace_back(0, 0);
    offsets.emplace_back(0 + n_new, 0);
    offsets.emplace_back(0, 0 + n_new);
    offsets.emplace_back(0 + n_new, 0 + n_new);

    all_work[valid_max_index].emplace_back(
        w.x + std::get<0>(offsets[0]), w.y + std::get<1>(offsets[0]), n_new);
    all_work[valid_max_index].emplace_back(
        w.x + std::get<0>(offsets[1]), w.y + std::get<1>(offsets[1]), n_new);

    all_work[min_index].emplace_back(w.x + std::get<0>(offsets[2]),
                                     w.y + std::get<1>(offsets[2]), n_new);
    all_work[min_index].emplace_back(w.x + std::get<0>(offsets[3]),
                                     w.y + std::get<1>(offsets[3]), n_new);

    total_work[valid_max_index] -= 2 * n_new * n_new;
    total_work[min_index] += 2 * n_new * n_new;
  }

  return all_work;
}

std::vector<double> static_improved::matrix_multiply() {
  hpx::cout << "using pseudodynamic distributed algorithm" << std::endl
            << hpx::flush;
  size_t num_localities = hpx::get_num_localities().get();
  std::vector<hpx::id_type> all_ids = hpx::find_all_localities();

  size_t compute_localities = num_localities;
  if (all_ids.size() > 1) {
    std::cout << "info: root node is not used for computation" << std::endl;
    compute_localities = num_localities - 1;
    hpx::id_type root_locality = hpx::find_root_locality();
    for (auto it = all_ids.begin(); it < all_ids.end(); it++) {
      if (*it == root_locality) {
        all_ids.erase(it);
        break;
      }
    }
    for (hpx::id_type &id : all_ids) {
      std::cout << "computing on id: " << id << std::endl;
    }
  }

  hpx::default_distribution_policy policy = hpx::default_layout(all_ids);

  // one multiplier per node, to avoid additional copies of A and B

  std::vector<hpx::components::client<multiplier>> multipliers =
      hpx::new_<hpx::components::client<multiplier>[]>(
          policy, compute_localities, N, A, B, transposed, block_input, verbose)
          .get();

  for (hpx::components::client<multiplier> &multiplier : multipliers) {
    uint32_t comp_locality =
        hpx::naming::get_locality_id_from_id(multiplier.get_id());
    multiplier.register_as("/multiplier#" + std::to_string(comp_locality),
                           false);
  }

  std::vector<hpx::future<void>> futures;

  std::vector<hpx::components::client<recursive>> recursives;

  std::vector<std::vector<matrix_multiply_work>> all_work =
      this->create_schedule(compute_localities);

  if (verbose >= 1) {
    this->print_schedule(all_work);
  }

  for (size_t repeat = 0; repeat < repetitions; repeat++) {
    for (size_t i = 0; i < all_work.size(); i++) {
      // transmit the work to the processing locality
      for (matrix_multiply_work &w : all_work[i]) {
        hpx::components::client<recursive> recursive =
            hpx::new_<hpx::components::client<multiply_components::recursive>>(
                all_ids[i], block_result, verbose);

        uint32_t comp_locality =
            hpx::naming::get_locality_id_from_id(recursive.get_id());
        recursive.register_as("/recursive#" + std::to_string(comp_locality),
                              false);

        hpx::future<std::vector<double>> f =
            hpx::async<recursive::distribute_recursively_action>(
                recursive.get_id(), w.x, w.y, w.N);
        hpx::future<void> g =
            f.then(hpx::util::unwrapped([=](std::vector<double> submatrix) {
              this->insert_submatrix(submatrix, w);
            }));

        futures.push_back(std::move(g));
        recursives.push_back(std::move(recursive));
      }
    }

    hpx::wait_all(futures);
  }
  return C;
}
}
