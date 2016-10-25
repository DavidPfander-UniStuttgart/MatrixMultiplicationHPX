////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 David Pfander
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <random>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <cassert>
#include <chrono>

#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include "matrix_multiply_naive.hpp"
#include "matrix_multiply_kernel_test.hpp"
#include "matrix_multiply_kernel_tiled.hpp"
#include "matrix_multiply_util.hpp"

std::vector<double> A;
std::vector<double> B;
std::vector<double> C;
std::uint64_t N;
std::string algorithm;
std::uint64_t verbose;
bool check;
bool transposed;
uint64_t block_input;
size_t small_block_size;
// initialized via program_options defaults

double duration;
uint64_t repetitions;

int main(int argc, char* argv[]) {
  // Configure application-specific options
  boost::program_options::options_description desc_commandline;

  desc_commandline.add_options()("n-value",
				 boost::program_options::value<std::uint64_t>()->default_value(4),
				 "n value for the square matrices, should be a power of 2")(
											    "repetitions",
											    boost::program_options::value<std::uint64_t>()->default_value(1),
											    "how often should the operation be repeated (for averaging timings)")(
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

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc_commandline), vm);
  boost::program_options::notify(vm);
  // extract command line argument
  N = vm["n-value"].as<std::uint64_t>();
  small_block_size = vm["small-block-size"].as<std::uint64_t>();
  verbose = vm["verbose"].as<uint64_t>();
  algorithm = vm["algorithm"].as<std::string>();
  check = vm["check"].as<bool>();
  transposed = vm["transposed"].as<bool>();
  block_input = vm["block-input"].as<uint64_t>();
  repetitions = vm["repetitions"].as<uint64_t>();

    
  // create matrices A, B
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  auto myRand = std::bind(distribution, generator);

  A.resize(N * N);
  std::generate(A.begin(), A.end(), myRand);

  if (verbose >= 2) {
    std::cout << "matrix A:" << std::endl;
    print_matrix_host(N, A);
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
      print_matrix_host(N, B);
    } else {
      print_matrix_transposed_host(N, B);
    }
  }
    
  if (algorithm.compare("kernel_test") == 0) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
    kernel_test::matrix_multiply_kernel_test m(N, A, B, transposed, repetitions, verbose);
    C = m.matrix_multiply();

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(t2 - t1).count();
    
    std::cout << "non-HPX [N = " << N << "] total time: " << duration << "s"
    	      << std::endl;
    std::cout << "non-HPX [N = " << N << "] average time per run: "
    	      << (duration / static_cast<double>(repetitions)) << "s (repetitions = " << repetitions
    	      << ")" << std::endl;

    if (verbose >= 2) {
      std::cout << "non-HPX matrix C:" << std::endl;
      print_matrix_host(N, C);
    }
  } else if (algorithm.compare("kernel_tiled") == 0) {
    // std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
    kernel_tiled::matrix_multiply_kernel_tiled m(N, A, B, transposed, repetitions, verbose);
    double duration_inner = 0.0;
    C = m.matrix_multiply(duration_inner);
    duration = duration_inner;

    // std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration<double>(t2 - t1).count();
    
    std::cout << "non-HPX [N = " << N << "] total time: " << duration << "s"
	      << std::endl;
    std::cout << "non-HPX [N = " << N << "] average time per run: "
	      << (duration / repetitions) << "s (repetitions = " << repetitions
	      << ")" << std::endl;

    if (verbose >= 2) {
      std::cout << "non-HPX matrix C:" << std::endl;
      print_matrix_host(N, C);
    }
  }

  double flops = 2 * static_cast<double>(N) * static_cast<double>(N)
    * static_cast<double>(N);
  double gflop = flops / 1E9;
  std::cout << "[N = " << N << "] performance: "
  	    << (repetitions * gflop / duration)
  	    << "Gflops (average across repetitions)" << std::endl;

  // hpx should now be shut down, can now use CPU for (fast) checking

  if (check) {
    if (repetitions > 1) {
      std::cout
	<< "info: repetitions > 1: checking only last iteration"
	<< std::endl;
    }

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
    std::vector<double> Cref;
    if (!transposed) {
      Cref = naive_matrix_multiply(N, A, B);
    } else {
      Cref = naive_matrix_multiply_transposed(N, A, B);
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    double duration_reference = std::chrono::duration<double>(t2 - t1).count();
    
    char const* fmt = "naive matMult took %1% [s]";
    std::cout << (boost::format(fmt) % duration_reference) << std::endl;
    std::cout << "[N = " << N << "] performance reference: "
	      << (gflop / duration_reference)
	      << " Gflops (reference implementation)" << std::endl;

    if (verbose >= 2) {
      std::cout << "matrix Cref:" << std::endl;
      print_matrix_host(N, Cref);
    }

    // compare solutions
    bool ok = std::equal(C.begin(), C.end(), Cref.begin(), Cref.end(),
			 [](double first, double second) {
			   if (std::abs(first - second) < 1E-10) {
			     return true;
			   } else {
			     return false;
			   }
			 });
    if (ok) {
      std::cout << "check passed" << std::endl;
    } else {
      std::cout << "error: check failed!" << std::endl;
    }

    if (verbose >= 2) {
      std::vector<double> diff_matrix(N * N);
      for (size_t k = 0; k < N * N; k++) {
	diff_matrix.at(k) = fabs(Cref.at(k) - C.at(k));
      }
      std::cout << "diff_matrix:" << std::endl;
      print_matrix_host(N, diff_matrix);
    }
  }
  return 0;
}
