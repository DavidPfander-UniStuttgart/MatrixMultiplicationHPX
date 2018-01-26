#include "autotune/autotune.hpp"
#include "autotune/parameter.hpp"
#include "autotune/tuners/bruteforce.hpp"
#include "autotune/tuners/bruteforce.hpp"
#include "autotune/tuners/full_neighborhood_search.hpp"
#include "autotune/tuners/line_search.hpp"
#include "autotune/tuners/monte_carlo.hpp"
#include "autotune/tuners/neighborhood_search.hpp"

#include "util/create_random_matrix.hpp"
#include "util/matrix_multiplication_exception.hpp"
#include "util/util.hpp"
#include "variants/combined.hpp"
#include "variants/kernel_tiled.hpp"
#include "variants/naive.hpp"

#include <functional>
#include <random>

#include <chrono>
#include <omp.h>

AUTOTUNE_KERNEL(uint64_t(), hardware_query_kernel,
                "src/variants/hardware_query_kernel")

std::ofstream tuner_duration_file;

int main(int argc, char **argv) {

  tuner_duration_file.open("tuner_duration.csv");
  tuner_duration_file << "tuner, duration" << std::endl;

  if (argc < 2) {
    std::cerr << "Error: no scenario name given!" << std::endl;
    return 1;
  } else if (argc > 2) {
    std::cerr << "Error: two many arguments given!" << std::endl;
    return 1;
  }

  // figure out native vector width
  auto &builder_hw_query =
      autotune::hardware_query_kernel.get_builder<cppjit::builder::gcc>();
  builder_hw_query.set_verbose(true);
  builder_hw_query.set_include_paths(
      "-IAutoTuneTMP/AutoTuneTMP_install/include -Isrc/variants/ "
      "-IAutoTuneTMP/Vc_install/include "
      "-IAutoTuneTMP/boost_install/include");
  builder_hw_query.set_cpp_flags(
      "-Wall -Wextra -std=c++17 -march=native -mtune=native "
      "-O3 -g -ffast-math -fopenmp -fPIC -fno-gnu-unique");
  builder_hw_query.set_link_flags("-shared -fno-gnu-unique");
  size_t native_vector_width = autotune::hardware_query_kernel();
  std::cout << "native_vector_width: " << native_vector_width << std::endl;

  std::string scenario_name(argv[1]);
  std::cout << "scenario_name: " << scenario_name << std::endl;

  std::uint64_t N = 64;
  // std::uint64_t N = 256;

  bool transposed = false;
  size_t repetitions = 5;
  bool verbose = false;

  // create matrices A, B>
  std::vector<double> A = util::create_random_matrix<double>(N);
  std::vector<double> B = util::create_random_matrix<double>(N);

  std::vector<double> C_reference;
  {
    kernel_tiled::kernel_tiled m_tiled(N, A, B, transposed, repetitions,
                                       verbose);
    std::cout << "calculating reference solution..." << std::flush;
    double duration_reference;
    C_reference = m_tiled.matrix_multiply(duration_reference);
    // C_reference = naive_matrix_multiply(N, A, B);
  }

  // if (!transposed) {
  //   C_reference = naive_matrix_multiply(N, A, B);
  // } else {
  //   C_reference = naive_matrix_multiply_transposed(N, A, B);
  // }
  std::cout << " done" << std::endl << std::flush;

  if (transposed) {
    throw util::matrix_multiplication_exception(
        "algorithm \"combined\" doens't allow B to be transposed");
  }
  combined::combined m(N, A, B, repetitions, verbose);

  double duration_kernel;

  autotune::combined_kernel.set_verbose(true);
  autotune::combined_kernel.set_kernel_duration_functor(
      [&duration_kernel]() { return duration_kernel; });

  auto &builder = autotune::combined_kernel.get_builder<cppjit::builder::gcc>();
  builder.set_verbose(true);

  builder.set_include_paths(
      "-IAutoTuneTMP/AutoTuneTMP_install/include -Isrc/variants/ "
      "-IAutoTuneTMP/Vc_install/include "
      "-IAutoTuneTMP/boost_install/include");
  builder.set_cpp_flags("-Wall -Wextra -std=c++17 -march=native -mtune=native "
                        "-O3 -g -ffast-math -fopenmp -fPIC -fno-gnu-unique");
  builder.set_link_flags("-shared -g -fno-gnu-unique");

  autotune::countable_continuous_parameter p1("X_REG", 5, 1, 1, 5);        // 5
  autotune::countable_continuous_parameter p2("Y_BASE_WIDTH", 2, 1, 1, 5); // 5
  /*autotune::countable_continuous_parameter p3("L1_X", 30, 5, 10, 40);      // 8
  autotune::countable_continuous_parameter p4("L1_Y", 32, 8, 8, 64);       // 8
  autotune::countable_continuous_parameter p5("L1_K_STEP", 32, 16, 16,
                                              128);                     // 8
  autotune::countable_continuous_parameter p6("L2_X", 60, 10, 20, 100); // 8
  autotune::countable_continuous_parameter p7("L2_Y", 64, 16, 16, 128); // 8
  autotune::countable_continuous_parameter p8("L2_K_STEP", 64, 32, 32,
                                              256); // 7*/
  autotune::countable_continuous_parameter p3("L1_X_FACTOR", 1, 1, 1, 40);      // 8
  autotune::countable_continuous_parameter p4("L1_Y_FACTOR", 1, 1, 1, 64);       // 8
  autotune::countable_continuous_parameter p5("L1_K_STEP", 32, 16, 16,
                                              128);                     // 8
  autotune::countable_continuous_parameter p6("L2_X_FACTOR", 1, 1, 1, 100); // 8
  autotune::countable_continuous_parameter p7("L2_Y_FACTOR", 1, 1, 1, 128); // 8
  autotune::countable_continuous_parameter p8("L2_K_FACTOR", 1, 1, 1, 8); // 7

  // autotune::fixed_set_parameter<std::string> p10("X_REG", {"5"}, false);
  // autotune::fixed_set_parameter<std::string> p11("Y_BASE_WIDTH", {"2"},
  // false);

  size_t openmp_threads = omp_get_max_threads();
  std::vector<size_t> thread_values;
  thread_values.push_back(openmp_threads);
  for (size_t i = 0; i < 1; i++) { // 4-way HT assumed max
    // for (size_t i = 0; i < 3; i++) {  // 4-way HT assumed max
    if (openmp_threads % 2 == 0) {
      openmp_threads /= 2;
      thread_values.push_back(openmp_threads);
    } else {
      break;
    }
  }
  autotune::fixed_set_parameter<size_t> p9("KERNEL_OMP_THREADS", thread_values);

  autotune::countable_set parameters_group_register;
  autotune::countable_set parameters_group_l1;
  autotune::countable_set parameters_group_l2;
  autotune::countable_set parameters_group_other;
  parameters_group_register.add_parameter(p1);
  parameters_group_register.add_parameter(p2);
  parameters_group_l1.add_parameter(p3);
  parameters_group_l1.add_parameter(p4);
  parameters_group_l1.add_parameter(p5);
  parameters_group_l2.add_parameter(p6);
  parameters_group_l2.add_parameter(p7);
  parameters_group_l2.add_parameter(p8);
  parameters_group_other.add_parameter(p9);

  autotune::randomizable_set randomizable_parameters_group_register;
  autotune::randomizable_set randomizable_parameters_group_l1;
  autotune::randomizable_set randomizable_parameters_group_l2;
  autotune::randomizable_set randomizable_parameters_group_other;
  randomizable_parameters_group_register.add_parameter(p1);
  randomizable_parameters_group_register.add_parameter(p2);
  randomizable_parameters_group_l1.add_parameter(p3);
  randomizable_parameters_group_l1.add_parameter(p4);
  randomizable_parameters_group_l1.add_parameter(p5);
  randomizable_parameters_group_l2.add_parameter(p6);
  randomizable_parameters_group_l2.add_parameter(p7);
  randomizable_parameters_group_l2.add_parameter(p8);
  randomizable_parameters_group_other.add_parameter(p9);

  autotune::combined_kernel.set_source_dir("src/variants/combined_kernel");
  
  autotune::combined_kernel.set_parameter_values(parameters_group_register);
  autotune::combined_kernel.set_parameter_values(parameters_group_l1);
  autotune::combined_kernel.set_parameter_values(parameters_group_l2);
  autotune::combined_kernel.set_parameter_values(parameters_group_other);

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

  auto precompile_validate_parameter_functor =
      [native_vector_width](autotune::parameter_value_set &parameters) -> bool {
    int64_t X_REG = stol(parameters["X_REG"]);
    int64_t Y_BASE_WIDTH = stol(parameters["Y_BASE_WIDTH"]);
    //int64_t L1_X = stol(parameters["L1_X"]);
    //int64_t L1_Y = stol(parameters["L1_Y"]);
    int64_t L1_K_STEP = stol(parameters["L1_K_STEP"]);
    //int64_t L2_X = stol(parameters["L2_X"]);
    //int64_t L2_Y = stol(parameters["L2_Y"]);
    //int64_t L2_K_STEP = stol(parameters["L2_K_STEP"]);
    
    int64_t L1_X_FACTOR = stol(parameters["L1_X_FACTOR"]);
    int64_t L1_Y_FACTOR = stol(parameters["L1_Y_FACTOR"]);
    int64_t L2_X_FACTOR = stol(parameters["L2_X_FACTOR"]);
    int64_t L2_Y_FACTOR = stol(parameters["L2_Y_FACTOR"]);
    int64_t L2_K_FACTOR = stol(parameters["L2_K_FACTOR"]);
    
    int64_t L1_X = (X_REG*L1_X_FACTOR);
    int64_t L2_X = (X_REG*L1_X_FACTOR*L2_X_FACTOR);
    int64_t L2_Y = (Y_BASE_WIDTH*L1_Y_FACTOR*L2_Y_FACTOR);
    int64_t L2_K_STEP = (L1_K_STEP*L2_K_FACTOR);
    
    const int64_t Y_REG = Y_BASE_WIDTH * native_vector_width;
    
    int64_t L1_Y = (Y_REG*L1_Y_FACTOR);
    int64_t L2_Y = (Y_REG*L1_Y_FACTOR*L2_Y_FACTOR);

    if (L1_X < X_REG) {
      std::cout << "error: L1_X < X_REG, L1_X too small" << std::endl;
      return false;
    }
    if (L2_X < L1_X) {
      std::cout << "error: L2_X < L1_X, L2_X too small" << std::endl;
      return false;
    }
    if (L1_Y < Y_REG) {
      std::cout << "error: L1_Y < Y_REG, L1_Y too small" << std::endl;
      return false;
    }
    if (L2_Y < L1_Y) {
      std::cout << "error: L2_Y < L1_Y, L2_Y too small" << std::endl;
      return false;
    }
    if (L2_K_STEP < L1_K_STEP) {
      std::cout << "error: L2_K_STEP < L1_K_STEP, L2_K_STEP too small"
                << std::endl;
      return false;
    }
    if (L1_X % X_REG != 0) {
      std::cout << "error: L1_X does not divide X_REG" << std::endl;
      return false;
    }
    if (L1_Y % Y_REG != 0) {
      std::cout << "error: L1_Y does not divide Y_REG" << std::endl;
      return false;
    }
    if (L2_X % L1_X != 0) {
      std::cout << "error: x direction blocking error: L2_X % L1_X != 0"
                << std::endl;
      return false;
    }
    if (L2_Y % L1_Y != 0) {
      std::cout << "error: y direction blocking error: L2_Y % L1_Y != 0"
                << std::endl;
      return false;
    }
    if (L2_K_STEP % L1_K_STEP != 0) {
      std::cout
          << "error: k direction blocking error: L2_K_STEP % L1_K_STEP != 0 "
          << std::endl;
      return false;
    }
    if (L1_X > 40) {
      std::cout << "error: L1_X to large" << std::endl;
      return false;
    }
    if (L1_Y > 64) {
      std::cout << "error: L1_Y to large" << std::endl;
      return false;
    }
    if (L2_X > 100) {
      std::cout << "error: L2_X to large" << std::endl;
      return false;
    }
    if (L2_Y > 128) {
      std::cout << "error: L2_Y to large" << std::endl;
      return false;
    }
    if (L2_K_STEP > 256) {
      std::cout << "error: L2_K_STEP to large" << std::endl;
      return false;
    }
      return true;
    };
  autotune::combined_kernel.set_precompile_validate_parameter_functor(
      precompile_validate_parameter_functor);

  /*auto parameter_group_l1_adjustment_functor =
      [native_vector_width](autotune::countable_set &parameters,
        autotune::parameter_value_set parameter_values) -> void {
    auto x_reg = stol(parameter_values["X_REG"]);
    auto y_base_width = stol(parameter_values["Y_BASE_WIDTH"]);
    auto &l1_x =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L1_X");
    auto &l1_y =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L1_Y");

    const double y_reg_value =
        y_base_width * native_vector_width;

    // register parameters are always correct, never changed

    l1_x.to_nearest_valid(x_reg);

    l1_y.to_nearest_valid(y_reg_value);
  };
  auto parameter_group_l2_adjustment_functor =
      [native_vector_width](autotune::countable_set &parameters,
        autotune::parameter_value_set parameter_values) -> void {
    auto l1_x = stol(parameter_values["L1_X"]);
    auto l1_y = stol(parameter_values["L1_Y"]);
    auto l1_k_step = stol(parameter_values["L1_K_STEP"]);
    auto &l2_x =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L2_X");
    auto &l2_y =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L2_Y");
    auto &l2_k_step =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L2_K_STEP");

    // register parameters are always correct, never changed

    l2_x.to_nearest_valid(l1_x);

    l2_y.to_nearest_valid(l1_y);

    l2_k_step.to_nearest_valid(l1_k_step);
  };

  auto parameter_group_l1_adjustment_functor_randomizable =
      [native_vector_width](autotune::randomizable_set &parameters,
        autotune::parameter_value_set parameter_values) -> void {
    auto x_reg = stol(parameter_values["X_REG"]);
    auto y_base_width = stol(parameter_values["Y_BASE_WIDTH"]);
    auto &l1_x =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L1_X");
    auto &l1_y =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L1_Y");

    const double y_reg_value =
        y_base_width * native_vector_width;

    // register parameters are always correct, never changed

    l1_x.to_nearest_valid(x_reg);

    l1_y.to_nearest_valid(y_reg_value);
  };
  auto parameter_group_l2_adjustment_functor_randomizable =
      [native_vector_width](autotune::randomizable_set &parameters,
        autotune::parameter_value_set parameter_values) -> void {
    auto l1_x = stol(parameter_values["L1_X"]);
    auto l1_y = stol(parameter_values["L1_Y"]);
    auto l1_k_step = stol(parameter_values["L1_K_STEP"]);
    auto &l2_x =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L2_X");
    auto &l2_y =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L2_Y");
    auto &l2_k_step =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L2_K_STEP");

    // register parameters are always correct, never changed

    l2_x.to_nearest_valid(l1_x);

    l2_y.to_nearest_valid(l1_y);

    l2_k_step.to_nearest_valid(l1_k_step);
  };*/

  // tune with line search
  {
    std::cout
        << "----------------- starting tuning with line search ------------ "
        << std::endl;
    size_t line_search_steps = 50;
    size_t restarts = 5;
    for (size_t restart = 0; restart < restarts; restart++) {
      std::cout << "restart: " << restart << std::endl;
      bool valid_start_found = false;
      autotune::parameter_value_set original_parameters = 
        autotune::combined_kernel.get_parameter_values();
      autotune::parameter_value_set parameter_values;
      while (!valid_start_found) {
        for (size_t parameter_index = 0; parameter_index < parameters_group_register.size();
             parameter_index++) {
          auto &p = parameters_group_register[parameter_index];
          p->set_random_value();
          parameter_values[p->get_name()] = p->get_value();
        }
        for (size_t parameter_index = 0; parameter_index < parameters_group_l1.size();
             parameter_index++) {
          auto &p = parameters_group_l1[parameter_index];
          p->set_random_value();
          parameter_values[p->get_name()] = p->get_value();
        }
        for (size_t parameter_index = 0; parameter_index < parameters_group_l2.size();
             parameter_index++) {
          auto &p = parameters_group_l2[parameter_index];
          p->set_random_value();
          parameter_values[p->get_name()] = p->get_value();
        }
        for (size_t parameter_index = 0; parameter_index < parameters_group_other.size();
             parameter_index++) {
          auto &p = parameters_group_other[parameter_index];
          p->set_random_value();
          parameter_values[p->get_name()] = p->get_value();
        }
        if (precompile_validate_parameter_functor(parameter_values)) {
          valid_start_found = true;
        }
      }
      autotune::combined_kernel.set_parameter_values(parameter_values);
      { // tune parameter group register
        autotune::tuners::line_search tuner(autotune::combined_kernel, parameters_group_register,
                                            line_search_steps, 1);
        tuner.set_verbose(true);
        tuner.set_write_measurement(scenario_name + "_line_search_register_" +
                                    std::to_string(restart));

        tuner.setup_test(test_result);

        autotune::countable_set optimal_parameters;
        {
          std::chrono::high_resolution_clock::time_point start =
              std::chrono::high_resolution_clock::now();
          optimal_parameters = tuner.tune(m.N_org, m.A_org, m.B_org,
                                          m.repetitions, duration_kernel);
          std::chrono::high_resolution_clock::time_point end =
              std::chrono::high_resolution_clock::now();
          double tuning_duration =
              std::chrono::duration<double>(end - start).count();
          tuner_duration_file << "line_search_register, " << tuning_duration << std::endl;
        }
        autotune::combined_kernel.set_parameter_values(optimal_parameters);
      }
      { // tune parameter group l1
        autotune::tuners::line_search tuner(autotune::combined_kernel, parameters_group_l1,
                                            line_search_steps, 1);
        //tuner.set_parameter_adjustment_functor(parameter_group_l1_adjustment_functor);
        tuner.set_verbose(true);
        tuner.set_write_measurement(scenario_name + "_line_search_l1_" +
                                    std::to_string(restart));

        tuner.setup_test(test_result);

        autotune::countable_set optimal_parameters;
        {
          std::chrono::high_resolution_clock::time_point start =
              std::chrono::high_resolution_clock::now();
          optimal_parameters = tuner.tune(m.N_org, m.A_org, m.B_org,
                                          m.repetitions, duration_kernel);
          std::chrono::high_resolution_clock::time_point end =
              std::chrono::high_resolution_clock::now();
          double tuning_duration =
              std::chrono::duration<double>(end - start).count();
          tuner_duration_file << "line_search_l1, " << tuning_duration << std::endl;
        }
        autotune::combined_kernel.set_parameter_values(optimal_parameters);
      }
      { // tune parameter group l2
        autotune::tuners::line_search tuner(autotune::combined_kernel, parameters_group_l2,
                                            line_search_steps, 1);
        //tuner.set_parameter_adjustment_functor(parameter_group_l2_adjustment_functor);
        tuner.set_verbose(true);
        tuner.set_write_measurement(scenario_name + "_line_search_l2_" +
                                    std::to_string(restart));

        tuner.setup_test(test_result);

        autotune::countable_set optimal_parameters;
        {
          std::chrono::high_resolution_clock::time_point start =
              std::chrono::high_resolution_clock::now();
          optimal_parameters = tuner.tune(m.N_org, m.A_org, m.B_org,
                                          m.repetitions, duration_kernel);
          std::chrono::high_resolution_clock::time_point end =
              std::chrono::high_resolution_clock::now();
          double tuning_duration =
              std::chrono::duration<double>(end - start).count();
          tuner_duration_file << "line_search_l2, " << tuning_duration << std::endl;
        }
        autotune::combined_kernel.set_parameter_values(optimal_parameters);
      }
      { // tune oarameter group other
        autotune::tuners::line_search tuner(autotune::combined_kernel, parameters_group_other,
                                            line_search_steps, 1);
        tuner.set_verbose(true);
        tuner.set_write_measurement(scenario_name + "_line_search_other_" +
                                    std::to_string(restart));

        tuner.setup_test(test_result);

        autotune::countable_set optimal_parameters;
        {
          std::chrono::high_resolution_clock::time_point start =
              std::chrono::high_resolution_clock::now();
          optimal_parameters = tuner.tune(m.N_org, m.A_org, m.B_org,
                                          m.repetitions, duration_kernel);
          std::chrono::high_resolution_clock::time_point end =
              std::chrono::high_resolution_clock::now();
          double tuning_duration =
              std::chrono::duration<double>(end - start).count();
          tuner_duration_file << "line_search_other, " << tuning_duration << std::endl;
        }
        autotune::combined_kernel.set_parameter_values(optimal_parameters);
      }

      std::cout << "----------------------- end tuning -----------------------"
                << std::endl;
      std::cout << "optimal parameter values (line search):" << std::endl;
      autotune::print_parameter_values(autotune::combined_kernel.get_parameter_values());
      autotune::combined_kernel.compile();

      std::vector<double> C = m.matrix_multiply(duration_kernel);
      bool test_ok = test_result(C);
      if (test_ok) {
        std::cout << "optimal parameters test ok!" << std::endl;
      } else {
        std::cout << "optimal parameters FAILED test!" << std::endl;
      }

      double flops = 2 * static_cast<double>(N) * static_cast<double>(N) *
                     static_cast<double>(N);
      double gflop = flops / 1E9;
      std::cout << "optimal inner_duration (line search): " << duration_kernel
                << std::endl;
      std::cout << "[N = " << N << "] performance: "
                << ((repetitions * gflop) / duration_kernel) << "GFLOPS"
                << std::endl;
                
      autotune::combined_kernel.set_parameter_values(original_parameters);
    }
    for (size_t parameter_index = 0; parameter_index < parameters_group_register.size();
         parameter_index++) {
      auto &p = parameters_group_register[parameter_index];
      p->set_initial();
    }
    for (size_t parameter_index = 0; parameter_index < parameters_group_l1.size();
         parameter_index++) {
      auto &p = parameters_group_l1[parameter_index];
      p->set_initial();
    }
    for (size_t parameter_index = 0; parameter_index < parameters_group_l2.size();
         parameter_index++) {
      auto &p = parameters_group_l2[parameter_index];
      p->set_initial();
    }
    for (size_t parameter_index = 0; parameter_index < parameters_group_other.size();
         parameter_index++) {
      auto &p = parameters_group_other[parameter_index];
      p->set_initial();
    }
  }

  // tune with neighborhood search
  {
    std::cout << "----------------- starting tuning with neighborhood search"
                 "-----------------"
              << std::endl;

    size_t restarts = 5;
    size_t search_steps = 50;
    for (size_t restart = 0; restart < restarts; restart++) {
      std::cout << "restart: " << restart << std::endl;
      bool valid_start_found = false;
      autotune::parameter_value_set original_parameters = 
        autotune::combined_kernel.get_parameter_values();
      autotune::parameter_value_set parameter_values;
      while (!valid_start_found) {
        for (size_t parameter_index = 0; parameter_index < parameters_group_register.size();
             parameter_index++) {
          auto &p = parameters_group_register[parameter_index];
          p->set_random_value();
          parameter_values[p->get_name()] = p->get_value();
        }
        for (size_t parameter_index = 0; parameter_index < parameters_group_l1.size();
             parameter_index++) {
          auto &p = parameters_group_l1[parameter_index];
          p->set_random_value();
          parameter_values[p->get_name()] = p->get_value();
        }
        for (size_t parameter_index = 0; parameter_index < parameters_group_l2.size();
             parameter_index++) {
          auto &p = parameters_group_l2[parameter_index];
          p->set_random_value();
          parameter_values[p->get_name()] = p->get_value();
        }
        for (size_t parameter_index = 0; parameter_index < parameters_group_other.size();
             parameter_index++) {
          auto &p = parameters_group_other[parameter_index];
          p->set_random_value();
          parameter_values[p->get_name()] = p->get_value();
        }
        if (precompile_validate_parameter_functor(parameter_values)) {
          valid_start_found = true;
        }
      }
      autotune::combined_kernel.set_parameter_values(parameter_values);
      { // tune parameter group register
        autotune::tuners::neighborhood_search tuner(autotune::combined_kernel,
                                                    parameters_group_register, search_steps);
        tuner.set_verbose(true);
        tuner.set_write_measurement(scenario_name + "_neighborhood_search_register_" +
                                    std::to_string(restart));
        tuner.setup_test(test_result);

        autotune::countable_set optimal_parameters;
        {
          std::chrono::high_resolution_clock::time_point start =
              std::chrono::high_resolution_clock::now();
          optimal_parameters = tuner.tune(m.N_org, m.A_org, m.B_org,
                                          m.repetitions, duration_kernel);
          std::chrono::high_resolution_clock::time_point end =
              std::chrono::high_resolution_clock::now();
          double tuning_duration =
              std::chrono::duration<double>(end - start).count();
          tuner_duration_file << "neighborhood_search_register, " << tuning_duration
                              << std::endl;
        }
        autotune::combined_kernel.set_parameter_values(optimal_parameters);
      }
      { // tune parameter group l1
        autotune::tuners::neighborhood_search tuner(autotune::combined_kernel,
                                                    parameters_group_l1, search_steps);
        //tuner.set_parameter_adjustment_functor(parameter_group_l1_adjustment_functor);
        tuner.set_verbose(true);
        tuner.set_write_measurement(scenario_name + "_neighborhood_search_l1_" +
                                    std::to_string(restart));
        tuner.setup_test(test_result);

        autotune::countable_set optimal_parameters;
        {
          std::chrono::high_resolution_clock::time_point start =
              std::chrono::high_resolution_clock::now();
          optimal_parameters = tuner.tune(m.N_org, m.A_org, m.B_org,
                                          m.repetitions, duration_kernel);
          std::chrono::high_resolution_clock::time_point end =
              std::chrono::high_resolution_clock::now();
          double tuning_duration =
              std::chrono::duration<double>(end - start).count();
          tuner_duration_file << "neighborhood_search_l1, " << tuning_duration
                              << std::endl;
        }
        autotune::combined_kernel.set_parameter_values(optimal_parameters);
      }
      { // tune parameter group l2
        autotune::tuners::neighborhood_search tuner(autotune::combined_kernel,
                                                    parameters_group_l2, search_steps);
        //tuner.set_parameter_adjustment_functor(parameter_group_l2_adjustment_functor);
        tuner.set_verbose(true);
        tuner.set_write_measurement(scenario_name + "_neighborhood_search_l2_" +
                                    std::to_string(restart));
        tuner.setup_test(test_result);

        autotune::countable_set optimal_parameters;
        {
          std::chrono::high_resolution_clock::time_point start =
              std::chrono::high_resolution_clock::now();
          optimal_parameters = tuner.tune(m.N_org, m.A_org, m.B_org,
                                          m.repetitions, duration_kernel);
          std::chrono::high_resolution_clock::time_point end =
              std::chrono::high_resolution_clock::now();
          double tuning_duration =
              std::chrono::duration<double>(end - start).count();
          tuner_duration_file << "neighborhood_search_l2, " << tuning_duration
                              << std::endl;
        }
        autotune::combined_kernel.set_parameter_values(optimal_parameters);
      }
      { // tune parameter group other
        autotune::tuners::neighborhood_search tuner(autotune::combined_kernel,
                                                    parameters_group_other, search_steps);
        tuner.set_verbose(true);
        tuner.set_write_measurement(scenario_name + "_neighborhood_search_other_" +
                                    std::to_string(restart));
        tuner.setup_test(test_result);

        autotune::countable_set optimal_parameters;
        {
          std::chrono::high_resolution_clock::time_point start =
              std::chrono::high_resolution_clock::now();
          optimal_parameters = tuner.tune(m.N_org, m.A_org, m.B_org,
                                          m.repetitions, duration_kernel);
          std::chrono::high_resolution_clock::time_point end =
              std::chrono::high_resolution_clock::now();
          double tuning_duration =
              std::chrono::duration<double>(end - start).count();
          tuner_duration_file << "neighborhood_search_other, " << tuning_duration
                              << std::endl;
        }
        autotune::combined_kernel.set_parameter_values(optimal_parameters);
      }
      
      std::cout << "----------------------- end tuning -----------------------"
                << std::endl;
      std::cout << "optimal parameter values (neighborhood search):"
                << std::endl;
      autotune::print_parameter_values(autotune::combined_kernel.get_parameter_values());
      autotune::combined_kernel.compile();

      std::vector<double> C = m.matrix_multiply(duration_kernel);
      bool test_ok = test_result(C);
      if (test_ok) {
        std::cout << "optimal parameters test ok!" << std::endl;
      } else {
        std::cout << "optimal parameters FAILED test!" << std::endl;
      }

      double flops = 2 * static_cast<double>(N) * static_cast<double>(N) *
                     static_cast<double>(N);
      double gflop = flops / 1E9;
      std::cout << "optimal inner_duration (neighborhood search): "
                << duration_kernel << std::endl;
      std::cout << "[N = " << N << "] performance: "
                << ((repetitions * gflop) / duration_kernel) << "GFLOPS"
                << std::endl;
                
      autotune::combined_kernel.set_parameter_values(original_parameters);
    }
    for (size_t parameter_index = 0; parameter_index < parameters_group_register.size();
         parameter_index++) {
      auto &p = parameters_group_register[parameter_index];
      p->set_initial();
    }
    for (size_t parameter_index = 0; parameter_index < parameters_group_l1.size();
         parameter_index++) {
      auto &p = parameters_group_l1[parameter_index];
      p->set_initial();
    }
    for (size_t parameter_index = 0; parameter_index < parameters_group_l2.size();
         parameter_index++) {
      auto &p = parameters_group_l2[parameter_index];
      p->set_initial();
    }
    for (size_t parameter_index = 0; parameter_index < parameters_group_other.size();
         parameter_index++) {
      auto &p = parameters_group_other[parameter_index];
      p->set_initial();
    }
  }

  // tune with full neighborhood search
  {
    std::cout
        << "----------------- starting tuning with full neighborhood search"
           "-----------------"
        << std::endl;

    size_t restarts = 5;
    size_t search_steps = 50;
    for (size_t restart = 0; restart < restarts; restart++) {
      std::cout << "restart: " << restart << std::endl;
      bool valid_start_found = false;
      autotune::parameter_value_set original_parameters = 
        autotune::combined_kernel.get_parameter_values();
      autotune::parameter_value_set parameter_values;
      while (!valid_start_found) {
        for (size_t parameter_index = 0; parameter_index < parameters_group_register.size();
             parameter_index++) {
          auto &p = parameters_group_register[parameter_index];
          p->set_random_value();
          parameter_values[p->get_name()] = p->get_value();
        }
        for (size_t parameter_index = 0; parameter_index < parameters_group_l1.size();
             parameter_index++) {
          auto &p = parameters_group_l1[parameter_index];
          p->set_random_value();
          parameter_values[p->get_name()] = p->get_value();
        }
        for (size_t parameter_index = 0; parameter_index < parameters_group_l2.size();
             parameter_index++) {
          auto &p = parameters_group_l2[parameter_index];
          p->set_random_value();
          parameter_values[p->get_name()] = p->get_value();
        }
        for (size_t parameter_index = 0; parameter_index < parameters_group_other.size();
             parameter_index++) {
          auto &p = parameters_group_other[parameter_index];
          p->set_random_value();
          parameter_values[p->get_name()] = p->get_value();
        }
        if (precompile_validate_parameter_functor(parameter_values)) {
          valid_start_found = true;
        }
      }
      std::cout << "checkpoint" << std::endl;
      autotune::combined_kernel.set_parameter_values(parameter_values);
      { // tune parameter group register
        autotune::tuners::full_neighborhood_search tuner(
            autotune::combined_kernel, parameters_group_register, search_steps);
        tuner.set_verbose(true);
        tuner.set_write_measurement(scenario_name + "_full_neighborhood_search_register_" +
                                    std::to_string(restart));
        tuner.setup_test(test_result);

        autotune::countable_set optimal_parameters;
        {
          std::chrono::high_resolution_clock::time_point start =
              std::chrono::high_resolution_clock::now();
          optimal_parameters = tuner.tune(m.N_org, m.A_org, m.B_org,
                                          m.repetitions, duration_kernel);
          std::chrono::high_resolution_clock::time_point end =
              std::chrono::high_resolution_clock::now();
          double tuning_duration =
              std::chrono::duration<double>(end - start).count();
          tuner_duration_file << "full_neighborhood_search_register, " << tuning_duration
                              << std::endl;
        }
        autotune::combined_kernel.set_parameter_values(optimal_parameters);
      }
      { // tune parameter group l1
        autotune::tuners::full_neighborhood_search tuner(
            autotune::combined_kernel, parameters_group_l1, search_steps);
        //tuner.set_parameter_adjustment_functor(parameter_group_l1_adjustment_functor);
        tuner.set_verbose(true);
        tuner.set_write_measurement(scenario_name + "_full_neighborhood_search_l1_" +
                                    std::to_string(restart));
        tuner.setup_test(test_result);

        autotune::countable_set optimal_parameters;
        {
          std::chrono::high_resolution_clock::time_point start =
              std::chrono::high_resolution_clock::now();
          optimal_parameters = tuner.tune(m.N_org, m.A_org, m.B_org,
                                          m.repetitions, duration_kernel);
          std::chrono::high_resolution_clock::time_point end =
              std::chrono::high_resolution_clock::now();
          double tuning_duration =
              std::chrono::duration<double>(end - start).count();
          tuner_duration_file << "full_neighborhood_search_l1, " << tuning_duration
                              << std::endl;
        }
        autotune::combined_kernel.set_parameter_values(optimal_parameters);
      }
      { // tune parameter group l2
        autotune::tuners::full_neighborhood_search tuner(
            autotune::combined_kernel, parameters_group_l2, search_steps);
        //tuner.set_parameter_adjustment_functor(parameter_group_l2_adjustment_functor);
        tuner.set_verbose(true);
        tuner.set_write_measurement(scenario_name + "_full_neighborhood_search_l2_" +
                                    std::to_string(restart));
        tuner.setup_test(test_result);

        autotune::countable_set optimal_parameters;
        {
          std::chrono::high_resolution_clock::time_point start =
              std::chrono::high_resolution_clock::now();
          optimal_parameters = tuner.tune(m.N_org, m.A_org, m.B_org,
                                          m.repetitions, duration_kernel);
          std::chrono::high_resolution_clock::time_point end =
              std::chrono::high_resolution_clock::now();
          double tuning_duration =
              std::chrono::duration<double>(end - start).count();
          tuner_duration_file << "full_neighborhood_search_l2, " << tuning_duration
                              << std::endl;
        }
        autotune::combined_kernel.set_parameter_values(optimal_parameters);
      }
      { // tune parameter group other
        autotune::tuners::full_neighborhood_search tuner(
            autotune::combined_kernel, parameters_group_other, search_steps);
        tuner.set_verbose(true);
        tuner.set_write_measurement(scenario_name + "_full_neighborhood_search_other_" +
                                    std::to_string(restart));
        tuner.setup_test(test_result);

        autotune::countable_set optimal_parameters;
        {
          std::chrono::high_resolution_clock::time_point start =
              std::chrono::high_resolution_clock::now();
          optimal_parameters = tuner.tune(m.N_org, m.A_org, m.B_org,
                                          m.repetitions, duration_kernel);
          std::chrono::high_resolution_clock::time_point end =
              std::chrono::high_resolution_clock::now();
          double tuning_duration =
              std::chrono::duration<double>(end - start).count();
          tuner_duration_file << "full_neighborhood_search_other, " << tuning_duration
                              << std::endl;
        }
        autotune::combined_kernel.set_parameter_values(optimal_parameters);
      }
      std::cout << "----------------------- end tuning -----------------------"
                << std::endl;
      std::cout << "optimal parameter values (full neighborhood search):"
                << std::endl;
      autotune::print_parameter_values(autotune::combined_kernel.get_parameter_values());
      autotune::combined_kernel.compile();

      std::vector<double> C = m.matrix_multiply(duration_kernel);
      bool test_ok = test_result(C);
      if (test_ok) {
        std::cout << "optimal parameters test ok!" << std::endl;
      } else {
        std::cout << "optimal parameters FAILED test!" << std::endl;
      }

      double flops = 2 * static_cast<double>(N) * static_cast<double>(N) *
                     static_cast<double>(N);
      double gflop = flops / 1E9;
      std::cout << "optimal inner_duration (full neighborhood search): "
                << duration_kernel << std::endl;
      std::cout << "[N = " << N << "] performance: "
                << ((repetitions * gflop) / duration_kernel) << "GFLOPS"
                << std::endl;
      
      autotune::combined_kernel.set_parameter_values(original_parameters);
    }
    for (size_t parameter_index = 0; parameter_index < parameters_group_register.size();
         parameter_index++) {
      auto &p = parameters_group_register[parameter_index];
      p->set_initial();
    }
    for (size_t parameter_index = 0; parameter_index < parameters_group_l1.size();
         parameter_index++) {
      auto &p = parameters_group_l1[parameter_index];
      p->set_initial();
    }
    for (size_t parameter_index = 0; parameter_index < parameters_group_l2.size();
         parameter_index++) {
      auto &p = parameters_group_l2[parameter_index];
      p->set_initial();
    }
    for (size_t parameter_index = 0; parameter_index < parameters_group_other.size();
         parameter_index++) {
      auto &p = parameters_group_other[parameter_index];
      p->set_initial();
    }
  }

  // // tune with bruteforce search
  // {
  //   std::cout << "----------------- starting tuning with bruteforce search
  //   "
  //                "------------ "
  //             << std::endl;
  //   autotune::tuners::bruteforce tuner(autotune::combined_kernel,
  //   parameters);
  //   tuner.set_parameter_adjustment_functor(parameter_adjustment_functor);
  //   tuner.set_verbose(true);
  //   tuner.set_write_measurement(scenario_name + "_bruteforce_search");

  //   tuner.setup_test(test_result);
  //   autotune::countable_set optimal_parameters =
  //       tuner.tune(m.N_org, m.A_org, m.B_org, m.repetitions,
  //                  duration_kernel);

  //   std::cout << "----------------------- end tuning
  //   -----------------------"
  //             << std::endl;
  //   std::cout << "optimal parameter values (bruteforce search):" <<
  //   std::endl;
  //   optimal_parameters.print_values();
  //   autotune::combined_kernel.set_parameter_values(optimal_parameters);
  //   autotune::combined_kernel.compile();

  //   std::vector<double> C = m.matrix_multiply(duration_kernel);
  //   bool test_ok = test_result(C);
  //   if (test_ok) {
  //     std::cout << "optimal parameters test ok!" << std::endl;
  //   } else {
  //     std::cout << "optimal parameters FAILED test!" << std::endl;
  //   }

  //   double flops = 2 * static_cast<double>(N) * static_cast<double>(N) *
  //                  static_cast<double>(N);
  //   double gflop = flops / 1E9;
  //   std::cout << "optimal inner_duration (bruteforce search): "
  //             << duration_kernel << std::endl;
  //   std::cout << "[N = " << N
  //             << "] performance: " << ((repetitions * gflop) /
  //             duration_kernel)
  //             << "GFLOPS" << std::endl;
  // }

  // tune with monte carlo search
  {
    std::cout << "----------------- starting tuning with monte_carlo search"
                 "------------ "
              << std::endl;
    autotune::parameter_value_set original_parameters = 
        autotune::combined_kernel.get_parameter_values();
    size_t search_steps = 500;
    { // tune parameter group register
      autotune::tuners::monte_carlo tuner(autotune::combined_kernel,
                                          randomizable_parameters_group_register, search_steps);
      tuner.set_verbose(true);
      tuner.set_write_measurement(scenario_name + "_monte_carlo_search_register_");
      tuner.setup_test(test_result);

      autotune::randomizable_set optimal_parameters;
      {
        std::chrono::high_resolution_clock::time_point start =
            std::chrono::high_resolution_clock::now();
        optimal_parameters =
            tuner.tune(m.N_org, m.A_org, m.B_org, m.repetitions, duration_kernel);
        std::chrono::high_resolution_clock::time_point end =
            std::chrono::high_resolution_clock::now();
        double tuning_duration =
            std::chrono::duration<double>(end - start).count();
        tuner_duration_file << "mone_carlo_search_register, " << tuning_duration
                            << std::endl;
      }
      autotune::combined_kernel.set_parameter_values(optimal_parameters);
    }
    { // tune parameter group l1
      autotune::tuners::monte_carlo tuner(autotune::combined_kernel,
                                          randomizable_parameters_group_l1, search_steps);
      tuner.set_verbose(true);
      tuner.set_write_measurement(scenario_name + "_monte_carlo_search_l1_");
      //tuner.set_parameter_adjustment_functor(
      //    parameter_group_l1_adjustment_functor_randomizable);
      tuner.setup_test(test_result);

      autotune::randomizable_set optimal_parameters;
      {
        std::chrono::high_resolution_clock::time_point start =
            std::chrono::high_resolution_clock::now();
        optimal_parameters =
            tuner.tune(m.N_org, m.A_org, m.B_org, m.repetitions, duration_kernel);
        std::chrono::high_resolution_clock::time_point end =
            std::chrono::high_resolution_clock::now();
        double tuning_duration =
            std::chrono::duration<double>(end - start).count();
        tuner_duration_file << "mone_carlo_search_l1, " << tuning_duration
                            << std::endl;
      }
      autotune::combined_kernel.set_parameter_values(optimal_parameters);
    }
    { // tune parameter group l2
      autotune::tuners::monte_carlo tuner(autotune::combined_kernel,
                                          randomizable_parameters_group_l2, search_steps);
      tuner.set_verbose(true);
      tuner.set_write_measurement(scenario_name + "_monte_carlo_search_l2_");
      //tuner.set_parameter_adjustment_functor(
      //    parameter_group_l2_adjustment_functor_randomizable);
      tuner.setup_test(test_result);

      autotune::randomizable_set optimal_parameters;
      {
        std::chrono::high_resolution_clock::time_point start =
            std::chrono::high_resolution_clock::now();
        optimal_parameters =
            tuner.tune(m.N_org, m.A_org, m.B_org, m.repetitions, duration_kernel);
        std::chrono::high_resolution_clock::time_point end =
            std::chrono::high_resolution_clock::now();
        double tuning_duration =
            std::chrono::duration<double>(end - start).count();
        tuner_duration_file << "mone_carlo_search_l2, " << tuning_duration
                            << std::endl;
      }
      autotune::combined_kernel.set_parameter_values(optimal_parameters);
    }
    { // tune parameter group other
      autotune::tuners::monte_carlo tuner(autotune::combined_kernel,
                                          randomizable_parameters_group_other, search_steps);
      tuner.set_verbose(true);
      tuner.set_write_measurement(scenario_name + "_monte_carlo_search_other_");
      tuner.setup_test(test_result);

      autotune::randomizable_set optimal_parameters;
      {
        std::chrono::high_resolution_clock::time_point start =
            std::chrono::high_resolution_clock::now();
        optimal_parameters =
            tuner.tune(m.N_org, m.A_org, m.B_org, m.repetitions, duration_kernel);
        std::chrono::high_resolution_clock::time_point end =
            std::chrono::high_resolution_clock::now();
        double tuning_duration =
            std::chrono::duration<double>(end - start).count();
        tuner_duration_file << "mone_carlo_search_other, " << tuning_duration
                            << std::endl;
      }
      autotune::combined_kernel.set_parameter_values(optimal_parameters);
    }

    std::cout << "----------------------- end tuning----------------------"
              << std::endl;
    std::cout << "optimal parameter values (monte carlo search):" << std::endl;
    autotune::print_parameter_values(autotune::combined_kernel.get_parameter_values());
    autotune::combined_kernel.compile();

    std::vector<double> C = m.matrix_multiply(duration_kernel);
    bool test_ok = test_result(C);
    if (test_ok) {
      std::cout << "optimal parameters test ok!" << std::endl;
    } else {
      std::cout << "optimal parameters FAILED test!" << std::endl;
    }

    double flops = 2 * static_cast<double>(N) * static_cast<double>(N) *
                   static_cast<double>(N);
    double gflop = flops / 1E9;
    std::cout << "optimal inner_duration (monte carlo search): "
              << duration_kernel << std::endl;
    std::cout << "[N = " << N
              << "] performance: " << ((repetitions * gflop) / duration_kernel)
              << "GFLOPS" << std::endl;
    
    autotune::combined_kernel.set_parameter_values(original_parameters);
  }
}
