#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>

#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include "util/matrix_multiplication_exception.hpp"
#include "util/util.hpp"
#include "variants/combined.hpp"
#include "variants/kernel_test.hpp"
#include "variants/kernel_tiled.hpp"
#include "variants/naive.hpp"

boost::program_options::options_description
    desc_commandline("Usage: matrix_multiply [options]");

int main(int argc, char *argv[]) {
  desc_commandline.add_options()(
      "n-value",
      boost::program_options::value<std::uint64_t>()->default_value(4ull),
      "n value for the square matrices, should be a power of 2, arbitrary "
      "sized "
      "square matrices work with some implementations")(
      "transposed", boost::program_options::value<bool>()->default_value(true),
      "use a transposed matrix for B")(
      "repetitions",
      boost::program_options::value<std::uint64_t>()->default_value(1),
      "how often should the operation be repeated (for averaging timings)")(
      "verbose", boost::program_options::value<uint64_t>()->default_value(0),
      "set to 1 for some status information, set to 2 more output")(
      "check", boost::program_options::value<bool>()->default_value(false),
      "check result against a naive and slow matrix-multiplication "
      "implementation")(
      "algorithm",
      boost::program_options::value<std::string>()->default_value(
          "naive"), // TODO: truncate
      "select algorithm: single, pseudodynamic, algorithms, looped, semi, "
      "combined, kernel_test, kernel_tiled")("help", "display help");

  boost::program_options::variables_map vm;
  boost::program_options::store(
      boost::program_options::parse_command_line(argc, argv, desc_commandline),
      vm);
  boost::program_options::notify(vm);

  if (vm.count("help")) {
    std::cout << desc_commandline << std::endl;
    return 0;
  }

  // extract command line argument
  uint64_t N = vm["n-value"].as<std::uint64_t>();
  uint64_t verbose = vm["verbose"].as<uint64_t>();
  std::string algorithm = vm["algorithm"].as<std::string>();
  bool check = vm["check"].as<bool>();
  bool transposed = vm["transposed"].as<bool>();
  uint64_t repetitions = vm["repetitions"].as<uint64_t>();

  // create matrices A, B
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  auto myRand = std::bind(distribution, generator);

  std::vector<double> A;
  A.resize(N * N);
  std::generate(A.begin(), A.end(), myRand);

  if (verbose >= 2) {
    std::cout << "matrix A:" << std::endl << std::flush;
    print_matrix(N, A);
  }

  std::vector<double> B;
  B.resize(N * N);

  if (!transposed) {
    for (uint64_t i = 0; i < N; i++) {
      for (uint64_t j = 0; j < N; j++) {
        if (i == j) {
          B.at(i * N + j) = 1.0;
        } else {
          B.at(i * N + j) = 0.0;
        }
      }
    }
  } else {
    for (uint64_t i = 0; i < N; i++) {
      for (uint64_t j = 0; j < N; j++) {
        if (i == j) {
          B.at(j * N + i) = 1.0;
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

  std::vector<double> C;
  double duration;
  if (algorithm.compare("kernel_test") == 0) {
    std::cout << "using algorithm: kernel_test" << std::endl;
    auto timer_start = std::chrono::high_resolution_clock::now();
    kernel_test::kernel_test m(N, A, B, transposed, repetitions, verbose);
    C = m.matrix_multiply();
    auto timer_stop = std::chrono::high_resolution_clock::now();
    duration = (timer_stop - timer_start).count();
  } else if (algorithm.compare("kernel_tiled") == 0) {
    std::cout << "using algorithm: kernel_tiled" << std::endl;
    kernel_tiled::kernel_tiled m(N, A, B, transposed, repetitions, verbose);
    C = m.matrix_multiply(duration);
  } else if (algorithm.compare("naive") == 0) {
    std::cout << "using algorithm: naive" << std::endl;
    auto timer_start = std::chrono::high_resolution_clock::now();
    if (!transposed) {
      C = naive_matrix_multiply(N, A, B);
    } else {
      C = naive_matrix_multiply_transposed(N, A, B);
    }
    auto timer_stop = std::chrono::high_resolution_clock::now();
    duration = (timer_stop - timer_start).count();
  } else if (algorithm.compare("combined") == 0) {
    combined::combined m(N, A, B, repetitions, verbose);
    C = m.matrix_multiply(duration);
  } else {
    std::cout << "\"" << algorithm << "\" not a valid algorithm" << std::endl;
    return 1;
  }

  if (verbose >= 2) {
    std::cout << "matrix C:" << std::endl;
    print_matrix_host(N, C);
  }

  double flops = 2 * static_cast<double>(N) * static_cast<double>(N) *
                 static_cast<double>(N);
  double gflop = flops / 1E9;
  std::cout << "[N = " << N
            << "] performance: " << (repetitions * gflop / duration)
            << "Gflops (average across repetitions)" << std::endl;

  // hpx should now be shut down, can now use CPU for (fast) checking
  if (check) {
    if (repetitions > 1) {
      std::cout << "info: repetitions > 1: checking only last iteration"
                << std::endl;
    }
    std::vector<double> Cref;
    if (!transposed) {
      Cref = naive_matrix_multiply(N, A, B);
    } else {
      Cref = naive_matrix_multiply_transposed(N, A, B);
    }
    // char const *fmt = "naive matMult took %1% [s]";
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
