#include "combined.hpp"

#include <chrono>

#include "index_iterator.hpp"
#include "util/util.hpp"

#include "autotune/tuners/countable_set.hpp"

#include <Vc/Vc>
using Vc::double_v;

AUTOTUNE_DEFINE_KERNEL(std::vector<double>(std::size_t, std::size_t,
                                           std::size_t, std::size_t,
                                           std::vector<double> &,
                                           std::vector<double> &, size_t,
                                           double &),
                       combined_kernel)

using namespace index_iterator;

namespace combined {

combined::combined(size_t N, std::vector<double> &A_org,
                   std::vector<double> &B_org, uint64_t repetitions,
                   uint64_t verbose)
    : N_org(N), A(A_org), B(B_org), repetitions(repetitions), verbose(verbose) {
  // verify_blocking_setup();

  // k direction padding
  size_t k_pad = L3_K_STEP - (N % L3_K_STEP);
  if (k_pad == L3_K_STEP) {
    k_pad = 0; // nothing to pad
  }
  size_t x_pad = L3_X - (N % L3_X);
  if (x_pad == L3_X) {
    x_pad = 0; // nothing to pad
  }
  size_t y_pad = L3_Y - (N % L3_Y);
  if (y_pad == L3_Y) {
    y_pad = 0; // nothing to pad
  }

  if (verbose >= 1) {
    std::cout << "matrix padding: x_pad = " << x_pad << ", y_pad = " << y_pad
              << ", k_pad = " << k_pad << std::endl;
  }

  X_size = N + x_pad;
  Y_size = N + y_pad;
  K_size = N + k_pad;

  if (verbose >= 1) {
    std::cout << "matrix dimensions for calculation: X = " << X_size
              << ", Y = " << Y_size << ", K = " << K_size << std::endl;
  }

  A = std::vector<double>(X_size * K_size);
  std::fill(A.begin(), A.end(), 0.0);
  for (size_t x = 0; x < N_org; x++) {
    for (size_t k = 0; k < N_org; k++) {
      A.at(x * K_size + k) = A_org.at(x * N + k);
    }
  }
  B = std::vector<double>(K_size * Y_size);
  std::fill(B.begin(), B.end(), 0.0);
  for (size_t y = 0; y < N_org; y++) {
    for (size_t k = 0; k < N_org; k++) {
      B.at(k * Y_size + y) = B_org.at(k * N + y);
    }
  }
}

combined::~combined() { autotune::combined_kernel.clear(); }

std::vector<double> combined::matrix_multiply(double &duration) {

  if (!autotune::combined_kernel.is_compiled()) {

    auto builder =
        autotune::combined_kernel.get_builder_as<cppjit::builder::gcc>();
    builder->set_verbose(true);
    builder->set_include_paths("-IAutoTuneTMP/AutoTuneTMP_install/include -Isrc/variants/ "
                               "-IVc_install/include "
                               "-IAutoTuneTMP/boost_install/include");
    builder->set_cpp_flags(
        "-Wall -Wextra -std=c++17 -march=native -mtune=native "
        "-O3 -g -ffast-math -fopenmp -fPIC -fno-gnu-unique");
    builder->set_link_flags("-shared -fno-gnu-unique");

    // max 2 L3 par set to 1024 (rest 512)
    // static parameters, not tuned
    // autotune::combined_kernel.add_parameter("L3_X", {"420"});
    // autotune::combined_kernel.add_parameter("L3_Y", {"256"});
    // autotune::combined_kernel.add_parameter("L3_K_STEP", {"256"});

    std::string L3_X_s = std::to_string(L3_X);
    std::string L3_Y_s = std::to_string(L3_Y);
    std::string L3_K_STEP_s = std::to_string(L3_K_STEP);

    autotune::countable_set parameters;

    autotune::fixed_set_parameter<std::string> p1("L3_X", {L3_X_s}, false);
    parameters.add_parameter(p1);
    autotune::fixed_set_parameter<std::string> p2("L3_Y", {L3_Y_s}, false);
    parameters.add_parameter(p2);
    autotune::fixed_set_parameter<std::string> p3("L3_K_STEP", {L3_K_STEP_s},
                                                  false);
    parameters.add_parameter(p3);

    autotune::fixed_set_parameter<std::string> p4("L2_X", {"70"}, false);
    parameters.add_parameter(p4);
    autotune::fixed_set_parameter<std::string> p5("L2_Y", {"64"}, false);
    parameters.add_parameter(p5);
    autotune::fixed_set_parameter<std::string> p6("L2_K_STEP", {"128"}, false);
    parameters.add_parameter(p6);

    autotune::fixed_set_parameter<std::string> p7("L1_X", {"35"}, false);
    parameters.add_parameter(p7);
    autotune::fixed_set_parameter<std::string> p8("L1_Y", {"16"}, false);
    parameters.add_parameter(p8);
    autotune::fixed_set_parameter<std::string> p9("L1_K_STEP", {"64"}, false);
    parameters.add_parameter(p9);

    // autotune::combined_kernel.add_parameter("L3_X", {L3_X_s});
    // autotune::combined_kernel.add_parameter("L3_Y", {L3_Y_s});
    // autotune::combined_kernel.add_parameter("L3_K_STEP", {L3_K_STEP_s});

    // autotune::combined_kernel.add_parameter("L2_X", {"70"});
    // autotune::combined_kernel.add_parameter("L2_Y", {"64"});
    // autotune::combined_kernel.add_parameter("L2_K_STEP", {"128"});
    // autotune::combined_kernel.add_parameter("L1_X", {"35"});
    // autotune::combined_kernel.add_parameter("L1_Y", {"16"});
    // autotune::combined_kernel.add_parameter("L1_K_STEP", {"64"});

    // std::vector<size_t> parameter_indices(
    //     autotune::combined_kernel.get_parameters().size(), 0);

    autotune::combined_kernel.set_parameter_values(parameters);

    autotune::combined_kernel.set_source_dir("src/variants/combined_kernel");

    // autotune::combined_kernel.create_parameter_file(parameter_indices);

    autotune::combined_kernel.compile();

    if (!autotune::combined_kernel.is_valid_parameter_combination()) {
      throw;
    }

    std::cout << "compile finished!" << std::endl;
  } else {
    std::cout << "kernel already compiled! skipping compilation step"
              << std::endl;
  }

  // autotune::combined_kernel.print_parameters();

  duration = 0.0;

  std::vector<double> C_return;
  C_return = autotune::combined_kernel(N_org, X_size, Y_size, K_size, A, B,
                                       repetitions, duration);

  double flops = 2 * static_cast<double>(X_size) * static_cast<double>(Y_size) *
                 static_cast<double>(K_size);
  double gflop = flops / 1E9;
  std::cout << "[X_size = " << X_size << ", Y_size = " << Y_size
            << ", K_size = " << K_size
            << "] inner performance: " << (repetitions * gflop / duration)
            << "Gflops (average across repetitions)" << std::endl;

  return C_return;
}
}
