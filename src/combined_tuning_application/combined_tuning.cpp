#include "autotune/autotune.hpp"
#include "autotune/parameter.hpp"
#include "autotune/tuners/bruteforce.hpp"
#include "autotune/tuners/line_search.hpp"

#include "util/create_random_matrix.hpp"
#include "util/matrix_multiplication_exception.hpp"
#include "util/util.hpp"
#include "variants/combined.hpp"
#include "variants/naive.hpp"

#include <functional>
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

  std::uint64_t N = 256;

  bool transposed = false;
  size_t repetitions = 2;
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

  auto builder =
      autotune::combined_kernel.get_builder_as<cppjit::builder::gcc>();
  builder->set_verbose(true);

  builder->set_include_paths(
      "-IAutoTuneTMP/AutoTuneTMP_install/include -Isrc/variants/ "
      "-IAutoTuneTMP/Vc_install/include "
      "-IAutoTuneTMP/boost_install/include");
  builder->set_cpp_flags("-Wall -Wextra -std=c++17 -march=native -mtune=native "
                         "-O3 -g -ffast-math -fopenmp -fPIC -fno-gnu-unique");
  builder->set_link_flags("-shared -g -fno-gnu-unique");

  autotune::countable_set parameters;

  autotune::fixed_set_parameter<std::string> p1(
      "L3_X", {std::to_string(combined::L3_X)}, false);
  autotune::fixed_set_parameter<std::string> p2(
      "L3_Y", {std::to_string(combined::L3_Y)}, false);
  autotune::fixed_set_parameter<std::string> p3(
      "L3_K_STEP", {std::to_string(combined::L3_K_STEP)}, false);

  // definitions from combined.hpp
  // constexpr uint64_t L3_X = 420;
  // constexpr uint64_t L3_Y = 256;
  // constexpr uint64_t L3_K_STEP = 256;

  // TODO: improvement: only L1 has to be divisible by matrix size?

  autotune::countable_continuous_parameter p4("L2_X", 50, 10, 40, 100);
  autotune::countable_continuous_parameter p5("L2_Y", 64, 2, 16, 128,
                                              std::multiplies<double>(),
                                              std::divides<double>());
  autotune::countable_continuous_parameter p6("L2_K_STEP", 64, 2, 32, 256,
                                              std::multiplies<double>(),
                                              std::divides<double>());
  autotune::countable_continuous_parameter p7("L1_X", 30, 5, 10, 40);
  autotune::countable_continuous_parameter p8(
      "L1_Y", 64, 2, 16, 64, std::multiplies<double>(), std::divides<double>());
  autotune::countable_continuous_parameter p9("L1_K_STEP", 8, 2, 4, 256,
                                              std::multiplies<double>(),
                                              std::divides<double>());
  parameters.add_parameter(p1);
  parameters.add_parameter(p2);
  parameters.add_parameter(p3);
  parameters.add_parameter(p4);
  parameters.add_parameter(p5);
  parameters.add_parameter(p6);
  parameters.add_parameter(p7);
  parameters.add_parameter(p8);
  parameters.add_parameter(p9);

  // autotune::fixed_set_parameter<std::string> p1("L3_X", {"210", "420"},
  // false);
  // parameters.add_parameter(p1);
  // autotune::fixed_set_parameter<std::string> p2("L3_Y", {"128", "256"},
  // false);
  // parameters.add_parameter(p2);
  // autotune::fixed_set_parameter<std::string> p3("L3_K_STEP", {"256"}, false);
  // parameters.add_parameter(p3);

  // autotune::fixed_set_parameter<std::string> p4(
  //     "L2_X", {"15", "35", "70", "140", "175"}, false);
  // parameters.add_parameter(p4);
  // autotune::fixed_set_parameter<std::string> p5(
  //     "L2_Y", {"16", "32", "64", "128", "256"}, false);
  // parameters.add_parameter(p5);
  // autotune::fixed_set_parameter<std::string> p6(
  //     "L2_K_STEP", {"32", "64", "128", "256", "512"}, false);
  // parameters.add_parameter(p6);

  // autotune::fixed_set_parameter<std::string> p7("L1_X", {"5", "10", "35",
  // "70"},
  //                                               false);
  // parameters.add_parameter(p7);
  // autotune::fixed_set_parameter<std::string> p8(
  //     "L1_Y", {"16", "32", "64", "128"}, false);
  // parameters.add_parameter(p8);
  // autotune::fixed_set_parameter<std::string> p9(
  //     "L1_K_STEP", {"1", "4", "8", "16", "32"}, false);
  // parameters.add_parameter(p9);

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
  size_t line_search_steps = 50;
  autotune::tuners::line_search tuner(autotune::combined_kernel, parameters,
                                      line_search_steps, 1);
  tuner.set_verbose(true);
  tuner.set_write_measurement(scenario_name + "_line_search");

  tuner.set_validate_parameters_functor(
      [](autotune::countable_set &ps) -> bool {
        if (ps.get_by_name("L1_X") < ps.get_by_name("X_REG")) {
          std::cout << "error: L1_X < X_REG, L1_X too small" << std::endl;
          return false;
        }
        if (ps.get_by_name("L1_Y") < ps.get_by_name("Y_REG")) {
          std::cout << "error: L1_Y < Y_REG, L1_Y too small" << std::endl;
          return false;
        }
        // if (L1_X % X_REG != 0) {
	if (ps.get_by_name("L1_X") < ps.get_by_name("Y_REG")) {
          std::cout << "error: L1_X does not divide X_REG" << std::endl;
          return false;
        }
        if (L1_Y % Y_REG != 0) {
          std::cout << "error: L1_Y does not divide Y_REG" << std::endl;
          return false;
        }
        if (!((L2_X % L1_X == 0) && (L3_X % L2_X == 0))) {
          // if (L2_X % L1_X != 0) {
          std::cout << "error: x direction blocking not set up correctly"
                    << std::endl;
          return false;
        }
        if (!((L2_Y % L1_Y == 0) && (L3_Y % L2_Y == 0))) {
          // if (L2_Y % L1_Y != 0) {
          std::cout << "error: y direction blocking not set up correctly"
                    << std::endl;
          return false;
        }
        if (!((L2_K_STEP % L1_K_STEP == 0) && (L3_K_STEP % L2_K_STEP == 0))) {
          // if (L2_K_STEP % L1_K_STEP != 0) {
          std::cout << "error: k direction blocking not set up correctly"
                    << std::endl;
          return false;
        }
        return true;
      });

  tuner.setup_test(test_result);
  autotune::countable_set optimal_parameters =
      tuner.tune(m.N_org, m.X_size, m.Y_size, m.K_size, m.A, m.B, m.repetitions,
                 tune_kernel_duration_temp);

  std::cout << "----------------------- end tuning -----------------------"
            << std::endl;
  std::cout << "optimal parameter values:" << std::endl;
  optimal_parameters.print_values();
  autotune::combined_kernel.set_parameter_values(optimal_parameters);

  // autotune::combined_kernel.create_parameter_file(optimal_parameters);

  autotune::combined_kernel.compile();

  double inner_duration;
  std::vector<double> C = m.matrix_multiply(inner_duration);
  bool test_ok = test_result(C);
  if (test_ok) {
    std::cout << "optimal parameters test ok!" << std::endl;
  } else {
    std::cout << "optimal parameters FAILED test!" << std::endl;
  }

  double flops = 2 * static_cast<double>(N) * static_cast<double>(N) *
                 static_cast<double>(N);
  double gflop = flops / 1E9;
  std::cout << "inner_duration: " << inner_duration << std::endl;
  std::cout << "[N = " << N
            << "] performance: " << ((repetitions * gflop) / inner_duration)
            << "GFLOPS" << std::endl;
}
