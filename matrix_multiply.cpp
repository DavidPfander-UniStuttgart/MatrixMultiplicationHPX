////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 David Pfander
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
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
#include "matrix_multiply_recursive.hpp"
#include "matrix_multiply_multiplier.hpp"
#include "matrix_multiply_static.hpp"
#include "matrix_multiply_static_improved.hpp"
#include "matrix_multiply_util.hpp"

std::vector<double> A;
std::vector<double> B;
std::vector<double> C;
std::uint64_t N;
std::uint64_t verbose;
bool check;
bool transposed;
// initialized via program_options defaults

double duration;
// to skip printing and checking on all other nodes
bool is_root_node;

int hpx_main(boost::program_options::variables_map& vm) {

	// extract command line argument
	N = vm["n-value"].as<std::uint64_t>();
	size_t small_block_size = vm["small-block-size"].as<std::uint64_t>();
	verbose = vm["verbose"].as<uint64_t>();
	std::string algorithm = vm["algorithm"].as<std::string>();
	check = vm["check"].as<bool>();
	transposed = vm["transposed"].as<bool>();
	uint64_t block_input = vm["block-input"].as<uint64_t>();
	
	is_root_node = hpx::find_here() == hpx::find_root_locality();

	// create matrices A, B
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0, 1.0);
	auto myRand = std::bind(distribution, generator);

	A.resize(N * N);
	std::generate(A.begin(), A.end(), myRand);

	if (verbose >= 2) {
		std::cout << "matrix A:" << std::endl;
		print_matrix(N, A);
	}

	B.resize(N * N);

	if (!transposed) {
		for (uint64_t i = 0; i < N; i++) {
			for (uint64_t j = 0; j < N; j++) {
				if (i == j) {
					B.at(i * N + j) = 2.0;
				} else {
					B.at(i * N + j) = 0.0;
				}
			}
		}
	} else {
		for (uint64_t i = 0; i < N; i++) {
			for (uint64_t j = 0; j < N; j++) {
				if (i == j) {
					B.at(j * N + i) = 2.0;
				} else {
					B.at(j * N + i) = 0.0;
				}
			}
		}
	}

	if (verbose >= 2) {
		std::cout << "matrix B:" << std::endl;
		if (!transposed) {
			print_matrix(N, B);
		} else {
			print_matrix_transposed(N, B);
		}
	}

	//    C.resize(n * n);

	// Keep track of the time required to execute.
	hpx::util::high_resolution_timer t;

	if (algorithm.compare("single") == 0) {
		hpx::cout << "using parallel single node algorithm" << std::endl
				<< hpx::flush;
		hpx::components::client<matrix_multiply_multiplier> multiplier =
				hpx::new_<hpx::components::client<matrix_multiply_multiplier>>(
						hpx::find_here(), N, A, B, transposed, block_input,
						verbose);
		uint32_t comp_locality_multiplier = hpx::naming::get_locality_id_from_id(
				multiplier.get_id());
		multiplier.register_as("/multiplier#" + std::to_string(comp_locality_multiplier));
		hpx::components::client<matrix_multiply_recursive> recursive =
				hpx::new_<hpx::components::client<matrix_multiply_recursive>>(
						hpx::find_here(), small_block_size, verbose);
		uint32_t comp_locality_recursive = hpx::naming::get_locality_id_from_id(
				recursive.get_id());
		recursive.register_as("/recursive#" + std::to_string(comp_locality_recursive));

		auto f = hpx::async<
				matrix_multiply_recursive::distribute_recursively_action>(
				recursive.get_id(), 0, 0, N);
		C = f.get();
	} else if (algorithm.compare("static") == 0) {
		matrix_multiply_static m(N, A, B, transposed, block_input,
				small_block_size, verbose);
		C = m.matrix_multiply();
	} else if (algorithm.compare("pseudodynamic") == 0) {
		std::uint64_t min_work_size = vm["min-work-size"].as<std::uint64_t>();
		std::uint64_t max_work_difference = vm["max-work-difference"].as<
				std::uint64_t>();
		double max_relative_work_difference =
				vm["max-relative-work-difference"].as<double>();

		matrix_multiply_static_improved m(N, A, B, transposed, block_input,
				small_block_size, min_work_size, max_work_difference,
				max_relative_work_difference, verbose);
		C = m.matrix_multiply();
	}

	duration = t.elapsed();
	char const* fmt = "matrixMultiply(n = %1%)\nelapsed time: %2% [s]";
	std::cout << (boost::format(fmt) % N % duration) << std::endl;

	if (verbose >= 2) {
		std::cout << "matrix C:" << std::endl;
		print_matrix(N, C);
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
					"single"), "select algorithm: single, static")(
			"min-work-size",
			boost::program_options::value<std::uint64_t>()->default_value(256),
			"for pseudodynamic scheduling, minimum work package size per node")(
			"max-work-difference",
			boost::program_options::value<std::uint64_t>()->default_value(
					10000),
			"for pseudodynamic scheduling, maximum tolerated load inbalance in matrix components assigned")(
			"max-relative-work-difference",
			boost::program_options::value<double>()->default_value(0.05),
			"for pseudodynamic scheduling, maximum relative tolerated load inbalance in matrix components assigned, in percent")(
			"transposed",
			boost::program_options::value<bool>()->default_value(true),
			"use a transposed matrix for B")("block-input",
			boost::program_options::value<uint64_t>()->default_value(0),
			"blocked application of the input matrices, set to 0 to disable");	

	// Initialize and run HPX
	int return_value = hpx::init(desc_commandline, argc, argv);

	if (is_root_node) {
	
	  //TODO: add multiple iterations
	  double flops = 2 * static_cast<double>(N) * static_cast<double>(N)
	    * static_cast<double>(N);
	  double gflop = flops / 1E9;
	  std::cout << (gflop / duration) << " Gflops" << std::endl;

	  // hpx should now be shut down, can now use CPU for (fast) checking

	  if (check) {
	    hpx::util::high_resolution_timer t2;
	    std::vector<double> Cref;
	    if (!transposed) {
	      Cref = naive_matrix_multiply(N, A, B);
	    } else {
	      Cref = naive_matrix_multiply_transposed(N, A, B);
	    }
	    char const* fmt = "naive matMult took %1% [s]";
	    double duration_reference = t2.elapsed();
	    std::cout << (boost::format(fmt) % duration_reference) << std::endl;
	    std::cout << (gflop / duration_reference) << " Gflops (reference implementation)" << std::endl;

	    if (verbose >= 2) {
	      std::cout << "matrix Cref:" << std::endl;
	      print_matrix(N, Cref);
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
	}
	return return_value;
}
