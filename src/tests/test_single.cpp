#define BOOST_TEST_DYN_LINK

#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>

#include <boost/test/unit_test.hpp>

// global test variables are defined here
#include "test_hpx_main.hpp"
#include "util/create_random_matrix.hpp"

#include "reference_kernels/kernel_test.hpp"
#include "reference_kernels/kernel_tiled.hpp"
#include "reference_kernels/naive.hpp"
#include "util/util.hpp"

BOOST_AUTO_TEST_SUITE(test_single)

BOOST_AUTO_TEST_CASE(random_matrices) {

	using namespace hpx_parameters;

	N = 512;

	A = util::create_random_matrix<double>(N);
	B = util::create_random_matrix<double>(N);
	C.resize(N * N);
	std::fill(C.begin(), C.end(), 0.0);
	C_reference.resize(N * N);
	std::fill(C_reference.begin(), C_reference.end(), 0.0);

	algorithm = "single";
	verbose = false;
	check = true;
	// actual transposition is irrelevant due to random matrices being used
	transposed = true;

	block_input = 128;
	block_result = 128;

	duration = 0.0; // write variable
	repetitions = 1;
	is_root_node = false;

	min_work_size = 0; // unused
	max_work_difference = 0; // unused
	max_relative_work_difference = 0.0; // unused

  // Initialize HPX, run hpx_main.
  hpx::start();

  // Wait for hpx::finalize being called.
  hpx::stop();

	if (!transposed) {
		C_reference = naive_matrix_multiply(N, A, B);
	} else {
		C_reference = naive_matrix_multiply_transposed(N, A, B);
	}

	// std::cout << "N: " << N << std::endl;
	// std::cout << "C:" << std::endl;
	// print_matrix_host(N, C);

	// std::cout << "C_reference:" << std::endl;
	// print_matrix_host(N, C_reference);

	for (size_t i = 0; i < N * N; i++) {
		BOOST_CHECK_CLOSE(C[i], C_reference[i], 1E-10);
	}
}

BOOST_AUTO_TEST_SUITE_END()
