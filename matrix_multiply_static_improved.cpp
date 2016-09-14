#include "matrix_multiply_static_improved.hpp"

#include <algorithm>

#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/lcos.hpp>

#include "matrix_multiply_recursive.hpp"
#include "matrix_multiply_multiplier.hpp"

void matrix_multiply_static_improved::insert_submatrix(
		const std::vector<double> &submatrix, const matrix_multiply_work &w) {
	for (size_t i = 0; i < w.N; i++) {
		for (size_t j = 0; j < w.N; j++) {
			// std::cout << (w.x + i) << ", " << (w.y + j) << " val: " << submatrix[i * w.N + j] << std::endl;
			C[(w.x + i) * N + (w.y + j)] = submatrix[i * w.N + j];
		}
	}
}

bool matrix_multiply_static_improved::fulfills_constraints(
		std::vector<uint64_t> &total_work) {
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

	if (max_work - min_work < max_work_difference) {
		return true;
	}
	if (static_cast<double>(max_work - min_work) / static_cast<double>(min_work)
			< max_relative_work_difference) {
		return true;
	}
	return false;
}

std::vector<std::vector<matrix_multiply_work>> matrix_multiply_static_improved::create_schedule(
		size_t num_localities) {
	// one slot per locality, stores the work to do
	std::vector<std::vector<matrix_multiply_work>> all_work(num_localities);

	// map: locality -> assigned components
	std::vector<uint64_t> total_work(num_localities);

	// initially all work is placed on the first locality
	all_work[0].emplace_back(0, 0, N);
	total_work[0] = N * N;

	size_t temp = 0;

	while (!this->fulfills_constraints(total_work)) {

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
		// give some work from the max element to the min element
		uint64_t max_work = total_work[0];
		size_t max_index = 0;
		uint64_t min_work = total_work[0];
		size_t min_index = 0;

		for (size_t i = 1; i < num_localities; i++) {
			if (total_work[i] > max_work) {
				max_work = total_work[i];
				max_index = i;
			}
			if (total_work[i] < min_work) {
				min_work = total_work[i];
				min_index = i;
			}
		}

		// will always succeed, do not have to skip first iteration
		size_t work_to_balance_index = 0;
		uint64_t smallest_work_difference = max_work - min_work;
		for (size_t j = 0; j < all_work[max_index].size(); j++) {
			matrix_multiply_work &w = all_work[max_index][j];
			// split work package between both locations
			uint64_t split_size = (w.N * w.N) / 2;
			uint64_t work_difference = (max_work - split_size)
					- (min_work + split_size);
			if (work_difference < smallest_work_difference) {
				smallest_work_difference = work_difference;
				work_to_balance_index = j;
			}
		}

		matrix_multiply_work w = all_work[max_index][work_to_balance_index];
		all_work[max_index].erase(
				all_work[max_index].begin() + work_to_balance_index);

		size_t n_new = w.N / 2;
		std::vector<std::tuple<size_t, size_t>> offsets = { { 0, 0 }, { 0
				+ n_new, 0 }, { 0, 0 + n_new }, { 0 + n_new, 0 + n_new } };

		all_work[max_index].emplace_back(w.x + std::get<0>(offsets[0]),
				w.y + std::get<1>(offsets[0]), n_new);
		all_work[max_index].emplace_back(w.x + std::get<0>(offsets[1]),
				w.y + std::get<1>(offsets[1]), n_new);

		all_work[min_index].emplace_back(w.x + std::get<0>(offsets[2]),
				w.y + std::get<1>(offsets[2]), n_new);
		all_work[min_index].emplace_back(w.x + std::get<0>(offsets[3]),
				w.y + std::get<1>(offsets[3]), n_new);

		temp += 1;
		if (temp > 5)
			break;
	}
	return all_work;
}

std::vector<double> matrix_multiply_static_improved::matrix_multiply() {
	hpx::cout << "using pseudodynamic distributed algorithm" << std::endl
			<< hpx::flush;
	size_t num_localities = hpx::get_num_localities().get();
	std::vector<hpx::id_type> all_ids = hpx::find_all_localities();
	hpx::default_distribution_policy policy = hpx::default_layout(all_ids);

// one multiplier per node, to avoid additional copies of A and B
	std::vector<hpx::components::client<matrix_multiply_multiplier>> multipliers =
			hpx::new_<hpx::components::client<matrix_multiply_multiplier>[]>(
					policy, num_localities, N, A, B, verbose).get();

// colocated
	for (hpx::components::client<matrix_multiply_multiplier> &multiplier : multipliers) {
		uint32_t comp_locality = hpx::naming::get_locality_id_from_id(
				multiplier.get_id());
		multiplier.register_as("/multiplier#" + std::to_string(comp_locality));
	}

	std::vector<hpx::future<void>> futures;

	std::vector<hpx::components::client<matrix_multiply_recursive>> recursives;

	std::vector<std::vector<matrix_multiply_work>> test1 =
			this->create_schedule(4);

	std::vector<std::vector<matrix_multiply_work>> all_work =
			this->create_schedule(num_localities);

	for (size_t i = 0; i < all_work.size(); i++) {
		// transmit the work to the processing locality
		for (matrix_multiply_work &w : all_work[i]) {
			hpx::components::client<matrix_multiply_recursive> recursive =
					hpx::new_<hpx::components::client<matrix_multiply_recursive>>(
							hpx::find_here(), small_block_size, verbose);
			hpx::future<std::vector<double>> f = hpx::async<
					matrix_multiply_recursive::distribute_recursively_action>(
					recursive.get_id(), w.x, w.y, w.N);
			hpx::future<void> g = f.then(
					hpx::util::unwrapped([=](std::vector<double> submatrix)
					{
						this->insert_submatrix(submatrix, w);
					}));
			futures.push_back(std::move(g));
			recursives.push_back(std::move(recursive));
		}
	}

	hpx::wait_all(futures);
	return C;
}
