////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 David Pfander
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/components.hpp>

#include <random>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <cassert>

#include <boost/format.hpp>

#include "matrix_multiply_naive.hpp"
#include "matrix_multiply_node.hpp"
#include "matrix_multiply_distributed.hpp"
#include "matrix_multiply_static.hpp"

//std::vector<double> A;
//std::vector<double> B;
//std::vector<double> C;
// initialized via program_options defaults
uint64_t verbose;
uint64_t small_block_size;
bool check;

int hpx_main(boost::program_options::variables_map& vm) {

	// extract command line argument
	std::uint64_t n = vm["n-value"].as<std::uint64_t>();
	small_block_size = vm["small-block-size"].as<std::uint64_t>();
	verbose = vm["verbose"].as<uint64_t>();
	check = vm["check"].as<bool>();
	std::string algorithm = vm["algorithm"].as<std::string>();

	std::vector<double> A;
	std::vector<double> B;
	std::vector<double> C;

	// create matrices A, B
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0, 1.0);
	auto myRand = std::bind(distribution, generator);

	A.resize(n * n);
	std::generate(A.begin(), A.end(), myRand);

	if (verbose >= 2) {
		std::cout << "matrix A:" << std::endl;
		print_matrix(n, A);
	}

	B.resize(n * n);

	for (uint64_t i = 0; i < n; i++) {
		for (uint64_t j = 0; j < n; j++) {
			if (i == j) {
				B.at(i * n + j) = 2.0;
			} else {
				B.at(i * n + j) = 0.0;
			}
		}
	}

	if (verbose >= 2) {
		std::cout << "matrix B:" << std::endl;
		print_matrix(n, B);
	}

//    C.resize(n * n);

// Keep track of the time required to execute.
	hpx::util::high_resolution_timer t;

	if (algorithm.compare("single") == 0) {
		hpx::cout << "using parallel single node algorithm" << std::endl << hpx::flush;
		hpx::components::client<matrix_multiply_node> multiplier = hpx::new_<
				hpx::components::client<matrix_multiply_node>>(hpx::find_here(),
				n, A, B);
		hpx::future<std::vector<double>> f = hpx::async<
				matrix_multiply_node::matrix_multiply_action>(
				multiplier.get_id(), 0, 0, n);
		C = f.get();
	} else if (algorithm.compare("static") == 0) {
	  matrix_multiply_static m;
	  C = m.matrix_multiply(n, A, B);
	}

	char const* fmt = "matrixMultiply(n = %1%)\nelapsed time: %2% [s]\n";
	std::cout << (boost::format(fmt) % n % t.elapsed());

	if (verbose >= 2) {
		std::cout << "matrix C:" << std::endl;
		print_matrix(n, C);
	}

	if (check) {
		hpx::util::high_resolution_timer t2;
		std::vector<double> Cref = naiveMatrixMultiply(n, A, B);
		char const* fmt = "naive matMult took %1% [s]\n";
		std::cout << (boost::format(fmt) % t2.elapsed());

		if (verbose >= 2) {
			std::cout << "matrix Cref:" << std::endl;
			print_matrix(n, Cref);
		}

		// compare solutions
		bool ok = std::equal(C.begin(), C.end(), Cref.begin(), Cref.end(),
				[](double first, double second) {
//							std::cout << "first: " << first << " second: " << second << std::endl;
					if (std::abs(first - second) < 1E-10) {
//								std::cout << "true" << std::endl;
						return true;
					} else {
//								std::cout << "false" << std::endl;
						return false;
					}
				});
		if (ok) {
			std::cout << "check passed" << std::endl;
		} else {
			std::cout << "error: check failed!" << std::endl;
		}
	}

	return hpx::finalize(); // Handles HPX shutdown
}

int main(int argc, char* argv[]) {
	// Configure application-specific options
	boost::program_options::options_description desc_commandline(
			"Usage: " HPX_APPLICATION_STRING " [options]");

	desc_commandline.add_options()("n-value",
			boost::program_options::value<std::uint64_t>()->default_value(4),
			"n value for the square matrices, should be a power of 2")(
			"small-block-size",
			boost::program_options::value<std::uint64_t>()->default_value(64),
			"cut-off value for smaller matrices to compute within a single thread")(
			"verbose",
			boost::program_options::value<uint64_t>()->default_value(0),
			"set to 1 for status information, set to 2 to additionally print the matrices")(
			"check",
			boost::program_options::value<bool>()->default_value(false),
			"check result against a naive and slow matrix-multiplication implementation")(
			"algorithm",
			boost::program_options::value<std::string>()->default_value(
					"single"), "select algorithm: single, static");

	// Initialize and run HPX
	return hpx::init(desc_commandline, argc, argv);
}
