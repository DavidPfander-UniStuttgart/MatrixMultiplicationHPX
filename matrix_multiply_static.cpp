#include "matrix_multiply_static.hpp"

#include "matrix_multiply_recursive.hpp"
#include "matrix_multiply_multiplier.hpp"

#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/lcos.hpp>

void matrix_multiply_static::insert_submatrix(const std::vector<double> &submatrix, const matrix_multiply_work &w) {
  for (size_t i = 0; i < w.N; i++) {
      for (size_t j = 0; j < w.N; j++) {
	// std::cout << (w.x + i) << ", " << (w.y + j) << " val: " << submatrix[i * w.N + j] << std::endl;
	C[(w.x + i) * N + (w.y + j)] = submatrix[i * w.N + j];
      }
  }
}

std::vector<matrix_multiply_work> matrix_multiply_static::create_work_packages(size_t num_localities) {
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

std::vector<double> matrix_multiply_static::matrix_multiply() {
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

  // one multiplier per node, to avoid additional copies of A and B
  std::vector<hpx::components::client<matrix_multiply_multiplier>> multipliers = hpx::new_<
    hpx::components::client<matrix_multiply_multiplier>[]>(policy, num_localities,
						     N, A, B, transposed, block_input, verbose).get();

  // colocated
  for (hpx::components::client<matrix_multiply_multiplier> &multiplier: multipliers) {
    uint32_t comp_locality = hpx::naming::get_locality_id_from_id(multiplier.get_id());
    multiplier.register_as("/multiplier#" + std::to_string(comp_locality));
  }

  // now place a recursive matrix splitter on every locality and give it a reference to its multiplier
  std::vector<hpx::components::client<matrix_multiply_recursive>> recursives =
    hpx::new_<hpx::components::client<matrix_multiply_recursive>[]>(policy, num_localities, small_block_size, verbose).get();

  std::vector<hpx::future<void>> futures;

  std::vector<matrix_multiply_work> work = this->create_work_packages(num_localities);

  std::vector<matrix_multiply_work>::iterator work_iterator = work.begin();		
  
  for (hpx::components::client<matrix_multiply_recursive> &recursive: recursives) {
    matrix_multiply_work &w = *work_iterator;
    
    hpx::future<std::vector<double>> f = hpx::async<
      matrix_multiply_recursive::distribute_recursively_action>(recursive.get_id(), w.x, w.y, w.N);
    // f.wait();
    hpx::future<void> g = f.then(hpx::util::unwrapped(
    				  [=](std::vector<double> submatrix)
    				  {
    				    this->insert_submatrix(submatrix, w);
    				  }));
    // implicitly moved
    futures.push_back(std::move(g));
    work_iterator++;
  }

  std::vector<hpx::components::client<matrix_multiply_recursive>> extra_recursives;
  // if either num_localities is not a power of 2 or num_localities is not divisible by 4
  // all additional work is done by locality 0!
  // Careful: Need additional components on node!
  while (work_iterator != work.end()) {
    if (verbose >= 1) {
      hpx::cout << "warning: extra work for first locality!" << std::endl << hpx::flush;
    }
    hpx::components::client<matrix_multiply_recursive> extra_recursive = hpx::new_<
      hpx::components::client<matrix_multiply_recursive>>(hpx::find_here(), small_block_size, verbose);    
    matrix_multiply_work &w = *work_iterator;
    hpx::future<std::vector<double>> f = hpx::async<
      matrix_multiply_recursive::distribute_recursively_action>(extra_recursive.get_id(), w.x, w.y, w.N);    
    hpx::future<void> g = f.then(hpx::util::unwrapped(
  						      [=](std::vector<double> submatrix)
  						      {
  							this->insert_submatrix(submatrix, w);
  						      }));
    futures.push_back(std::move(g));
    extra_recursives.push_back(std::move(extra_recursive));
    work_iterator++;
  }

  hpx::wait_all(futures);
  // for (hpx::future<std::vector<double>> &f: fs) {
  //   std::vector<double> C_small = f.get();
  //   this->insert_submatrix(C, C_small, N, w);
  // }
  // hpx::cout << "C.s: " << C.size() << std::endl << hpx::flush;
  return C;
}
