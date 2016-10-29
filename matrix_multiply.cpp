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
#include "matrix_multiply_static_improved.hpp"
#include "matrix_multiply_algorithms.hpp"
#include "matrix_multiply_looped.hpp"
#include "matrix_multiply_semi.hpp"
#include "matrix_multiply_combined.hpp"
#include "matrix_multiply_kernel_test.hpp"
#include "matrix_multiply_kernel_tiled.hpp"
#include "matrix_multiply_util.hpp"

boost::program_options::options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

bool display_help = false;

std::vector<double> A;
std::vector<double> B;
std::vector<double> C;
std::uint64_t N;
std::string algorithm;
std::uint64_t verbose;
bool check;
bool transposed;
uint64_t block_input;
size_t block_result;
// initialized via program_options defaults

double duration;
uint64_t repetitions;
// to skip printing and checking on all other nodes
bool is_root_node;
bool non_hpx_algorithm = false;

int hpx_main(boost::program_options::variables_map& vm) {

    // extract command line argument
    N = vm["n-value"].as<std::uint64_t>();
    block_result = vm["block-result"].as<std::uint64_t>();
    verbose = vm["verbose"].as<uint64_t>();
    algorithm = vm["algorithm"].as<std::string>();
    check = vm["check"].as<bool>();
    transposed = vm["transposed"].as<bool>();
    block_input = vm["block-input"].as<uint64_t>();
    repetitions = vm["repetitions"].as<uint64_t>();

    is_root_node = hpx::find_here() == hpx::find_root_locality();

    if (vm.count("help")) {
        display_help = true;
        hpx::cout << "in here" << std::endl << hpx::flush;
        if (is_root_node) {
            hpx::cout << "in here2" << std::endl << hpx::flush;
            hpx::cout << desc_commandline << std::endl << hpx::flush;
        }
        return hpx::finalize();
    }

    // create matrices A, B
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    auto myRand = std::bind(distribution, generator);

    A.resize(N * N);
    std::generate(A.begin(), A.end(), myRand);

    if (verbose >= 2) {
        hpx::cout << "matrix A:" << std::endl << hpx::flush;
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
        hpx::cout << "matrix B:" << std::endl << hpx::flush;
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
        uint32_t comp_locality_multiplier =
                hpx::naming::get_locality_id_from_id(multiplier.get_id());
        multiplier.register_as(
                "/multiplier#" + std::to_string(comp_locality_multiplier),
                false);
        hpx::components::client<matrix_multiply_recursive> recursive =
                hpx::new_<hpx::components::client<matrix_multiply_recursive>>(
                        hpx::find_here(), block_result, verbose);
        uint32_t comp_locality_recursive = hpx::naming::get_locality_id_from_id(
                recursive.get_id());
        recursive.register_as(
                "/recursive#" + std::to_string(comp_locality_recursive), false);
        for (size_t repeat = 0; repeat < repetitions; repeat++) {
            auto f = hpx::async<
                    matrix_multiply_recursive::distribute_recursively_action>(
                    recursive.get_id(), 0, 0, N);
            C = f.get();
        }
    } else if (algorithm.compare("pseudodynamic") == 0) {
        std::uint64_t min_work_size = vm["min-work-size"].as<std::uint64_t>();
        std::uint64_t max_work_difference = vm["max-work-difference"].as<
                std::uint64_t>();
        double max_relative_work_difference =
                vm["max-relative-work-difference"].as<double>();

        matrix_multiply_static_improved m(N, A, B, transposed, block_input,
                block_result, min_work_size, max_work_difference,
                max_relative_work_difference, repetitions, verbose);
        C = m.matrix_multiply();
    } else if (algorithm.compare("algorithms") == 0) {
        algorithms::matrix_multiply_algorithms m(N, A, B, transposed,
                block_input, block_result, repetitions, verbose);
        C = m.matrix_multiply();
    } else if (algorithm.compare("looped") == 0) {
        looped::matrix_multiply_looped m(N, A, B, transposed, block_result,
                block_input, repetitions, verbose);
        C = m.matrix_multiply();
    } else if (algorithm.compare("semi") == 0) {
        semi::matrix_multiply_semi m(N, A, B, transposed, block_result,
                block_input, repetitions, verbose);
        C = m.matrix_multiply();
    } else if (algorithm.compare("combined") == 0) {
        combined::matrix_multiply_combined m(N, A, B, transposed, block_result,
                block_input, repetitions, verbose);
	double inner_duration;
        C = m.matrix_multiply(inner_duration);
    } else {
	non_hpx_algorithm = true;
        return hpx::finalize(); // Handles HPX shutdown
    }

    duration = t.elapsed();
    hpx::cout << "[N = " << N << "] total time: " << duration << "s"
            << std::endl << hpx::flush;
    hpx::cout << "[N = " << N << "] average time per run: "
            << (duration / repetitions) << "s (repetitions = " << repetitions
            << ")" << std::endl << hpx::flush;

    if (verbose >= 2) {
        hpx::cout << "matrix C:" << std::endl << hpx::flush;
        print_matrix(N, C);
    }

    return hpx::finalize(); // Handles HPX shutdown
}

int main(int argc, char* argv[]) {
// Configure application-specific options
//    boost::program_options::options_description desc_commandline(
//            "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("n-value",
            boost::program_options::value<std::uint64_t>()->default_value(4),
            "n value for the square matrices, should be a power of 2, arbitrary sized square matrices work with some implementations")(
            "repetitions",
            boost::program_options::value<std::uint64_t>()->default_value(1),
            "how often should the operation be repeated (for averaging timings)")(
            "block-input",
            boost::program_options::value<uint64_t>()->default_value(1),
            "blocked application of the input matrices, set to 0 to disable")(
            "block-result",
            boost::program_options::value<std::uint64_t>()->default_value(1),
            "square block of the input matrix to be processed by a single thread")(
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
            "use a transposed matrix for B")("help", "display help");

// Initialize and run HPX
    int return_value = hpx::init(desc_commandline, argc, argv);

    if (display_help) {
        return return_value;
    }

    if (algorithm.compare("kernel_test") == 0) {
        hpx::util::high_resolution_timer t;
        kernel_test::matrix_multiply_kernel_test m(N, A, B, transposed,
                repetitions, verbose);
        C = m.matrix_multiply();

        duration = t.elapsed();
        std::cout << "non-HPX [N = " << N << "] total time: " << duration << "s"
                << std::endl;
        std::cout << "non-HPX [N = " << N << "] average time per run: "
                << (duration / repetitions) << "s (repetitions = "
                << repetitions << ")" << std::endl;

        if (verbose >= 2) {
            std::cout << "non-HPX matrix C:" << std::endl;
            print_matrix_host(N, C);
        }
    } else if (algorithm.compare("kernel_tiled") == 0) {
        kernel_tiled::matrix_multiply_kernel_tiled m(N, A, B, transposed,
                repetitions, verbose);
        C = m.matrix_multiply(duration);

        std::cout << "non-HPX [N = " << N << "] total time: " << duration << "s"
                << std::endl;
        std::cout << "non-HPX [N = " << N << "] average time per run: "
                << (duration / repetitions) << "s (repetitions = "
                << repetitions << ")" << std::endl;

        if (verbose >= 2) {
            std::cout << "non-HPX matrix C:" << std::endl;
            print_matrix_host(N, C);
        }
    } else {
      if (non_hpx_algorithm) {
	std::cout << "\"" << algorithm << "\" not a valid algorithm" << std::endl;
	return 1;
      }
    }

    if (is_root_node) {

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

            if (verbose >= 2) {
                std::vector<double> diff_matrix(N * N);
                for (size_t k = 0; k < N * N; k++) {
                    diff_matrix.at(k) = fabs(Cref.at(k) - C.at(k));
                }
                std::cout << "diff_matrix:" << std::endl;
                print_matrix_host(N, diff_matrix);
            }
        }
    }
    return return_value;
}
