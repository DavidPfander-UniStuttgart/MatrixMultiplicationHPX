#include "matrix_multiply_static.hpp"

#include "matrix_multiply_node.hpp"

#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>

void matrix_multiply_static::insert_submatrix(std::vector<double> &C, std::vector<double> &submatrix, size_t N, matrix_multiply_work &w) {
  for (size_t i = 0; i < w.N; i++) {
      for (size_t j = 0; j < w.N; j++) {
	// std::cout << (w.x + i) << ", " << (w.y + j) << " val: " << submatrix[i * w.N + j] << std::endl;
	C[(w.x + i) * N + (w.y + j)] = submatrix[i * w.N + j];
      }
  }
}

std::vector<matrix_multiply_work> matrix_multiply_static::create_work_packages(size_t N, size_t num_localities) {
  std::vector<matrix_multiply_work> work;
  work.reserve(num_localities);
  
  work.emplace_back(0, 0, N);
  // try to create work for every locality
  while (work.size() < num_localities) {
    matrix_multiply_work w = work[0];
    work.erase(work.begin());

    size_t n_new = w.N / 2;

    std::vector<std::tuple<size_t, size_t>> offsets =
      {{ 0, 0 }, {0 + n_new, 0 }, { 0, 0 + n_new }, { 0 + n_new, 0 + n_new } };

    for (auto offset: offsets) {
      work.emplace_back(w.x + std::get<0>(offset), w.y + std::get<1>(offset), n_new);
    }

    std::cout << "-----------------------" << std::endl;
    for (auto &w: work) {
      std::cout << "x: " << w.x << " y: " << w.y << " n: " << w.N << std::endl;
    }
    
  }
  return work;
}

std::vector<double> matrix_multiply_static::matrix_multiply(size_t N, std::vector<double> &A, std::vector<double> &B) {
  hpx::cout << "using static distributed algorithm" << std::endl << hpx::flush;
  size_t num_localities = hpx::get_num_localities().get();
  double exponent = std::log2(num_localities);
  if (exponent != std::trunc(exponent)) {
    hpx::cout << "warning: number of localities is not a power of 2!" << std::endl << hpx::flush;
  }
  if (num_localities % 4 != 0) {
    hpx::cout << "warning: number of localities is not divisible by 4!" << std::endl << hpx::flush;
  }
  std::vector<hpx::id_type> all_ids = hpx::find_all_localities();
  hpx::default_distribution_policy policy = hpx::default_layout(all_ids);
  std::vector<hpx::components::client<matrix_multiply_node>> multipliers = hpx::new_<
    hpx::components::client<matrix_multiply_node>[]>(policy, num_localities,
							    N, A, B).get();
  std::vector<hpx::future<std::vector<double>>> fs(num_localities);

  std::vector<matrix_multiply_work> work = this->create_work_packages(N, num_localities);

  std::vector<matrix_multiply_work>::iterator work_iterator = work.begin();
		
  // TODO: private constructor to avoid illegal copy (if reference in next statement is removed)?

  //		hpx::future<std::vector<double>> f = hpx::async<
  //				matrix_multiply_distributed::matrix_multiply_action>(
  //				multipliers[0].get_id(), 0, 0, N);
  //		f.get();

  std::vector<double> C(N * N);
  
  for (hpx::components::client<matrix_multiply_node> &multiplier: multipliers) {
    matrix_multiply_work &w = *work_iterator;
    hpx::future<std::vector<double>> f = hpx::async<
      matrix_multiply_node::matrix_multiply_action>(multiplier.get_id(), w.x, w.y, w.N);
    std::vector<double> C_small = f.get();
    this->insert_submatrix(C, C_small, N, w);
    work_iterator++;
    
    //			fs.push_back(std::move(f));
  }

  // if either num_localities is not a power of 2 or num_localities is not divisible by 4
  // all additional work is done by locality 0!
  while (work_iterator != work.end()) {
    matrix_multiply_work &w = *work_iterator;
    hpx::future<std::vector<double>> f = hpx::async<
      matrix_multiply_node::matrix_multiply_action>(multipliers[0].get_id(), w.x, w.y, w.N);

    std::vector<double> C_small = f.get();
    this->insert_submatrix(C, C_small, N, w);	
    work_iterator++;
  }
  
  //		hpx::when_all(fs);
  return C;
}
