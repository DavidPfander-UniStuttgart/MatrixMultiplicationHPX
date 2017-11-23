#include "autotune/autotune.hpp"
#include "autotune/parameter.hpp"
#include "autotune/tuners/bruteforce.hpp"
#include "autotune/tuners/line_search.hpp"

#include "util/create_random_matrix.hpp"
#include "util/matrix_multiplication_exception.hpp"
#include "util/util.hpp"
#include "variants/combined.hpp"
#include "variants/naive.hpp"

#include <random>

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cerr << "Error: no scenario name given!" << std::endl;
    return 1;
  } else if (argc > 2) {
    std::cerr << "Error: two many arguments given!" << std::endl;
    return 1;
  }

  std::string scenario_name(argv[1]);
  std::cout << "scenario_name: " << scenario_name << std::endl;

  std::uint64_t N = 16;

  bool transposed = false;
  size_t repetitions = 7;
  bool verbose = false;

  // create matrices A, B>
  std::vector<double> A = util::create_random_matrix<double>(N);
  std::vector<double> B = util::create_random_matrix<double>(N);

  std::vector<double> C_reference;
  std::cout << "calculating reference solution..." << std::flush;
  if (!transposed) {
    C_reference = naive_matrix_multiply(N, A, B);
  } else {
    C_reference = naive_matrix_multiply_transposed(N, A, B);
  }
  std::cout << " done" << std::endl << std::flush;

  if (transposed) {
    throw util::matrix_multiplication_exception(
        "algorithm \"combined\" doens't allow B to be transposed");
  }
  combined::combined m(N, A, B, repetitions, verbose);

  autotune::combined_kernel.set_verbose(true);

  autotune::combined_kernel.set_write_measurement(scenario_name +
                                                  "_line_search");

  auto builder =
      autotune::combined_kernel.get_builder_as<cppjit::builder::gcc>();
  builder->set_verbose(true);

  builder->set_include_paths("-IAutoTuneTMP/src -Isrc/variants/ "
                             "-IVc_install/include "
                             "-Iboost_install/include");
  builder->set_cpp_flags("-Wall -Wextra -std=c++17 -march=native -mtune=native "
                         "-O3 -g -ffast-math -fopenmp -fPIC");
  builder->set_link_flags("-shared -g");

  // autotune::combined_kernel.add_parameter("L3_X", {"210", "420"});
  // autotune::combined_kernel.add_parameter("L3_Y", {"128", "256"});
  // autotune::combined_kernel.add_parameter("L3_K_STEP", {"256"});
  // autotune::combined_kernel.add_parameter("L2_X",
  //                                         {"15", "35", "70", "140", "175"});
  // autotune::combined_kernel.add_parameter("L2_Y",
  //                                         {"16", "32", "64", "128", "256"});
  // autotune::combined_kernel.add_parameter("L2_K_STEP",
  //                                         {"32", "64", "128", "256", "512"});
  // autotune::combined_kernel.add_parameter("L1_X", {"5", "10", "35", "70"});
  // autotune::combined_kernel.add_parameter("L1_Y", {"16", "32", "64", "128"});
  // autotune::combined_kernel.add_parameter("L1_K_STEP",
  //                                         {"1", "4", "8", "16", "32"});

  autotune::combined_kernel.add_parameter("L3_X", {"420"});
  autotune::combined_kernel.add_parameter("L3_Y", {"256"});
  autotune::combined_kernel.add_parameter("L3_K_STEP", {"256"});

  autotune::combined_kernel.add_parameter("L2_X", {"70"});
  autotune::combined_kernel.add_parameter("L2_Y", {"64"});
  autotune::combined_kernel.add_parameter("L2_K_STEP", {"128"});
  autotune::combined_kernel.add_parameter("L1_X", {"35"});
  autotune::combined_kernel.add_parameter("L1_Y", {"16"});
  autotune::combined_kernel.add_parameter("L1_K_STEP", {"64"});

  autotune::combined_kernel.set_source_dir("src/variants/combined_kernel");

  double tune_kernel_duration_temp;

  std::function<bool(const std::vector<double> &C)> test_result =
      [&C_reference, N](const std::vector<double> &C) -> bool {
    for (size_t i = 0; i < N * N; i++) {
      double threshold = 1E-8;
      if (fabs(C[i] - C_reference[i]) >= threshold) {
        std::cout << "test error C: " << C[i] << " C_ref: " << C_reference[i]
                  << " i: " << i << " (threshold: " << threshold << ")"
                  << std::endl;
        return false;
      }
    }
    return true;
  };

  std::cout
      << "----------------------- starting tuning  -----------------------"
      << std::endl;
  std::vector<size_t> line_search_initial_guess = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  size_t line_search_steps = 1;
  autotune::tuners::line_search<decltype(autotune::combined_kernel)> tuner(
      autotune::combined_kernel, line_search_steps, 1,
      line_search_initial_guess);

  tuner.setup_test(test_result);
  std::vector<size_t> optimal_parameter_indices =
      tuner.tune(m.N_org, m.X_size, m.Y_size, m.K_size, m.A, m.B, m.repetitions,
                 tune_kernel_duration_temp);

  std::cout << "----------------------- end tuning -----------------------"
            << std::endl;
  std::cout << "optimal parameter values:" << std::endl;
  autotune::combined_kernel.print_values(optimal_parameter_indices);

  autotune::combined_kernel.create_parameter_file(optimal_parameter_indices);

  autotune::combined_kernel.compile();

  double inner_duration;
  std::vector<double> C = m.matrix_multiply(inner_duration);
  bool test_ok = test_result(C);
  if (test_ok) {
    std::cout << "optimal parameters test ok!" << std::endl;
  } else {
    std::cout << "optimal parameters FAILED test!" << std::endl;
  }
}
