#include "combined.hpp"

#include <chrono>

#include "index_iterator.hpp"
#include "util/util.hpp"

#include "autotune/tuners/countable_set.hpp"

#include <Vc/Vc>
using Vc::double_v;

#include <omp.h>

AUTOTUNE_DEFINE_KERNEL(std::vector<double>(std::size_t, std::vector<double> &,
                                           std::vector<double> &, size_t, double &),
                       combined_kernel, "src/variants/combined_kernel")

using namespace index_iterator;

namespace combined {

combined::combined(size_t N, std::vector<double> &A_org, std::vector<double> &B_org,
                   uint64_t repetitions, uint64_t verbose)
    : N_org(N), A_org(A_org), B_org(B_org), repetitions(repetitions), verbose(verbose) {
  autotune::combined_kernel.set_verbose(verbose);
}

combined::~combined() {
  // autotune::combined_kernel.clear();
}

std::vector<double> combined::matrix_multiply(double &duration) {
  if (!autotune::combined_kernel.is_compiled()) {
    auto &builder = autotune::combined_kernel.get_builder<cppjit::builder::gcc>();
    builder.set_do_cleanup(false);
    // builder.set_verbose(true);
    builder.set_include_paths(
        "-IAutoTuneTMP/AutoTuneTMP_install/include -Isrc/variants/ "
        "-IAutoTuneTMP/Vc_install/include "
        "-IAutoTuneTMP/boost_install/include");
    builder.set_cpp_flags(
        "-Wall -Wextra -std=c++17 -march=native -mtune=native "
        "-O3 -g -ffast-math -fopenmp -fPIC -fno-gnu-unique");
    builder.set_link_flags("-shared -fno-gnu-unique");

    autotune::countable_set parameters;
    autotune::fixed_set_parameter<std::string> p4("L2_X", {"80"}, false);
    autotune::fixed_set_parameter<std::string> p5("L2_Y", {"128"}, false);
    autotune::fixed_set_parameter<std::string> p6("L2_K_STEP", {"64"}, false);
    autotune::fixed_set_parameter<std::string> p7("L1_X", {"20"}, false);
    autotune::fixed_set_parameter<std::string> p8("L1_Y", {"16"}, false);
    autotune::fixed_set_parameter<std::string> p9("L1_K_STEP", {"64"}, false);
    autotune::fixed_set_parameter<std::string> p10("X_REG", {"5"}, false);
    autotune::fixed_set_parameter<std::string> p11("Y_BASE_WIDTH", {"2"}, false);
    size_t openmp_threads = omp_get_max_threads();
    autotune::fixed_set_parameter<size_t> p12("KERNEL_OMP_THREADS", {openmp_threads});
    // autotune::fixed_set_parameter<size_t> p10("KERNEL_OMP_THREADS", {1});

    parameters.add_parameter(p4);
    parameters.add_parameter(p5);
    parameters.add_parameter(p6);
    parameters.add_parameter(p7);
    parameters.add_parameter(p8);
    parameters.add_parameter(p9);
    parameters.add_parameter(p10);
    parameters.add_parameter(p11);
    parameters.add_parameter(p12);

    autotune::combined_kernel.set_parameter_values(parameters);
    autotune::combined_kernel.compile();

    if (!autotune::combined_kernel.is_valid_parameter_combination()) {
      throw;
    }

    std::cout << "compile finished!" << std::endl;
  } else {
    std::cout << "kernel already compiled! skipping compilation step" << std::endl;
  }

  // autotune::combined_kernel.print_parameters();

  duration = 0.0;

  std::vector<double> C_return;
  // C_return = autotune::combined_kernel(N_org, X_size, Y_size, K_size, A, B,
  //                                      repetitions, duration);
  C_return = autotune::combined_kernel(N_org, A_org, B_org, repetitions, duration);

  // double flops = 2 * static_cast<double>(X_size) *
  // static_cast<double>(Y_size) *
  //                static_cast<double>(K_size);
  // double gflop = flops / 1E9;
  // std::cout << "[X_size = " << X_size << ", Y_size = " << Y_size
  //           << ", K_size = " << K_size
  //           << "] inner performance: " << (repetitions * gflop / duration)
  //           << "Gflops (average across repetitions)" << std::endl;

  return C_return;
}
}  // namespace combined
