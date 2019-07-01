#include "autotune/autotune.hpp"
#include "autotune/parameter.hpp"
#include "autotune/tuners/bruteforce.hpp"
#include "autotune/tuners/full_neighborhood_search.hpp"
#include "autotune/tuners/greedy_neighborhood_search.hpp"
#include "autotune/tuners/group_tuner.hpp"
#include "autotune/tuners/line_search.hpp"
#include "autotune/tuners/monte_carlo.hpp"
#include "autotune/tuners/neighborhood_search.hpp"
#include "autotune/tuners/parallel_line_search.hpp"

#include "util/create_random_matrix.hpp"
#include "util/matrix_multiplication_exception.hpp"
#include "util/util.hpp"
#include "variants/combined.hpp"
#include "variants/kernel_tiled.hpp"
#include "variants/naive.hpp"

#include <functional>
#include <random>

#ifdef WITH_LIKWID
#include <likwid.h>
#endif
#include <chrono>
#include <omp.h>

// #define WITH_LIBLIKWID // controlled by cmake

AUTOTUNE_KERNEL(uint64_t(), hardware_query_kernel,
                "src/variants/hardware_query_kernel")

std::ofstream tuner_duration_file;

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

  uint64_t l1_size_bytes = 0;
  uint64_t l2_size_bytes = 0;
#ifdef WITH_LIKWID
  // use likwid to query some hardware information
  {
    int err = topology_init();
    if (err < 0) {
      std::cerr << "Unable to initialize likwid" << std::endl;
      return 1;
    } else {
      std::cout << "info: using likwid to query hardware information"
                << std::endl;
    }
    // CpuInfo_t contains global information like name, CPU family, ...
    CpuInfo_t info = get_cpuInfo();
    std::cout << "using cpu name: " << info->name << std::endl;
    // CpuTopology_t contains information about the topology of the CPUs.
    CpuTopology_t topo = get_cpuTopology();
    for (size_t i = 0; i < topo->numCacheLevels; i++) {
      std::cout << "level: " << topo->cacheLevels[i].level << std::endl;
      std::cout << "size: " << topo->cacheLevels[i].size << std::endl;
      if (topo->cacheLevels[i].level == 1) {
        l1_size_bytes = topo->cacheLevels[i].size;
      } else if (topo->cacheLevels[i].level == 2) {
        l2_size_bytes = topo->cacheLevels[i].size;
      }
    }
  }
#else
  std::cout << "Not using likwid, querying internal database for hardware specs"
            << std::endl;
  if (scenario_name.compare("6700k") == 0) {
    l1_size_bytes = 32 * 1024;
    l2_size_bytes = 256 * 1024;
  } else if (scenario_name.compare("4300U") == 0) {
    l1_size_bytes = 32 * 1024;
    l2_size_bytes = 256 * 1024;
  } else if (scenario_name.compare("xeonsilver") == 0) {
    l1_size_bytes = 32 * 1024;
    l2_size_bytes = 1024 * 1024;
  } else if (scenario_name.compare("knl") == 0) {
    l1_size_bytes = 32 * 1024;
    l2_size_bytes = 512 * 1024;
  } else if (scenario_name.compare("epyc") == 0) {
    l1_size_bytes = 32 * 1024;
    l2_size_bytes = 512 * 1024;
  } else if (scenario_name.compare("element") == 0) {
    l1_size_bytes = 32 * 1024;
    l2_size_bytes = 512 * 1024;
  } else {
    std::cerr
        << "error: platform hardware unknown and not compiled with liblikwid, "
           "aborting..."
        << std::endl;
    return 1;
  }
  std::cout << "level: " << 1 << std::endl;
  std::cout << "size: " << l1_size_bytes << std::endl;
  std::cout << "level: " << 2 << std::endl;
  std::cout << "size: " << l2_size_bytes << std::endl;
#endif

  // figure out native vector width
  auto &builder_hw_query =
      autotune::hardware_query_kernel.get_builder<cppjit::builder::gcc>();
  // builder_hw_query.set_verbose(true);
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

  tuner_duration_file.open(scenario_name + "_tuner_duration.csv");
  tuner_duration_file << "tuner, duration" << std::endl;

  bool transposed = false;
  bool verbose = false;

  /////////////// end boilerplate //////////////

  // create matrices A, B>
  std::vector<double> A_4096 = util::create_random_matrix<double>(4096);
  std::vector<double> B_4096 = util::create_random_matrix<double>(4096);

  std::vector<double> C_reference_4096;
  {
    kernel_tiled::kernel_tiled m_tiled(4096, A_4096, B_4096, transposed, 1,
                                       verbose);
    std::cout << "calculating reference solution..." << std::flush;
    double duration_reference;
    C_reference_4096 = m_tiled.matrix_multiply(duration_reference);
  }
  std::cout << " done (4096)" << std::endl << std::flush;

  // std::vector<double> A_2048 = util::create_random_matrix<double>(2048);
  // std::vector<double> B_2048 = util::create_random_matrix<double>(2048);

  // std::vector<double> C_reference_2048;
  // {
  //   kernel_tiled::kernel_tiled m_tiled(2048, A_2048, B_2048, transposed, 1,
  //   verbose); std::cout << "calculating reference solution..." << std::flush;
  //   double duration_reference;
  //   C_reference_2048 = m_tiled.matrix_multiply(duration_reference);
  // }
  // std::cout << " done (2048)" << std::endl << std::flush;

  if (transposed) {
    throw util::matrix_multiplication_exception(
        "algorithm \"combined\" doens't allow B to be transposed");
  }
  double duration_kernel;
  double gflops_kernel;

  autotune::combined_kernel.set_verbose(true);
  autotune::combined_kernel.set_kernel_duration_functor(
      [&duration_kernel]() { return duration_kernel; });

  auto &builder = autotune::combined_kernel.get_builder<cppjit::builder::gcc>();
  // builder.set_verbose(true);

  builder.set_include_paths(
      "-IAutoTuneTMP/AutoTuneTMP_install/include -Isrc/variants/ "
      "-IAutoTuneTMP/Vc_install/include "
      "-IAutoTuneTMP/boost_install/include -IAutoTuneTMP/likwid/src/includes");
  builder.set_cpp_flags("-Wall -Wextra -std=c++17 -march=native -mtune=native "
                        "-O3 -g -ffast-math -fopenmp -fPIC -fno-gnu-unique");
  builder.set_library_paths("-LAutoTuneTMP/libkwid");
  builder.set_link_flags("-shared -g -fno-gnu-unique");
  builder.set_libraries("-lnuma -llikwid");

  autotune::fixed_set_parameter<int> p0a("KERNEL_NUMA",
                                         {0, 1}); // 0 == none, 1 == copy
  autotune::fixed_set_parameter<int> p0b("KERNEL_SCHEDULE",
                                         {0, 1}); // 0==static, 1==dynamic
  autotune::countable_continuous_parameter p1("X_REG", 5, 1, 1, 5);        // 5
  autotune::countable_continuous_parameter p2("Y_BASE_WIDTH", 2, 1, 1, 5); // 5
  autotune::countable_continuous_parameter p3("L1_X", 30, 5, 10, 40);      // 8
  autotune::countable_continuous_parameter p4("L1_Y", 32, 8, 8, 64);       // 8
  autotune::countable_continuous_parameter p5("L1_K_STEP", 32, 16, 16,
                                              128);                     // 8
  autotune::countable_continuous_parameter p6("L2_X", 60, 10, 20, 100); // 8
  autotune::countable_continuous_parameter p7("L2_Y", 64, 16, 16, 128); // 8
  autotune::countable_continuous_parameter p8("L2_K_STEP", 64, 32, 32,
                                              256); // 7

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

  autotune::countable_set parameters;
  parameters.add_parameter(p0a);
  parameters.add_parameter(p0b);
  parameters.add_parameter(p1);
  parameters.add_parameter(p2);
  parameters.add_parameter(p3);
  parameters.add_parameter(p4);
  parameters.add_parameter(p5);
  parameters.add_parameter(p6);
  parameters.add_parameter(p7);
  parameters.add_parameter(p8);
  parameters.add_parameter(p9);

  autotune::combined_kernel.set_source_dir("src/variants/combined_kernel");

  auto precompile_validate_parameter_functor =
      [native_vector_width, l2_size_bytes,
       l1_size_bytes](autotune::parameter_value_set &parameters) -> bool {
    int64_t X_REG = stol(parameters["X_REG"]);
    int64_t Y_BASE_WIDTH = stol(parameters["Y_BASE_WIDTH"]);
    int64_t L1_X = stol(parameters["L1_X"]);
    int64_t L1_Y = stol(parameters["L1_Y"]);
    int64_t L1_K_STEP = stol(parameters["L1_K_STEP"]);
    int64_t L2_X = stol(parameters["L2_X"]);
    int64_t L2_Y = stol(parameters["L2_Y"]);
    int64_t L2_K_STEP = stol(parameters["L2_K_STEP"]);

    const int64_t Y_REG = Y_BASE_WIDTH * native_vector_width;

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

    size_t l2_memory = (L2_X * L2_K_STEP + L2_K_STEP * L2_Y + L2_X * L2_Y) * 8;
    if (l2_memory > l2_size_bytes) {
      std::cout << "rejected by l2 cache size requirement" << std::endl;
      return false;
    }
    size_t l1_memory = (L1_X * L1_K_STEP + L1_K_STEP * L1_Y + L1_X * L1_Y) * 8;
    if (l1_memory > l1_size_bytes) {
      std::cout << "rejected by l1 cache size requirement" << std::endl;
      return false;
    }

    return true;
  };
  autotune::combined_kernel.set_precompile_validate_parameter_functor(
      precompile_validate_parameter_functor);

  auto parameter_adjustment_functor =
      [native_vector_width](autotune::countable_set &parameters) -> void {
    auto &x_reg =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "X_REG");
    auto &y_base_width =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "Y_BASE_WIDTH");
    auto &l1_x =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L1_X");
    auto &l1_y =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L1_Y");
    auto &l1_k_step =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L1_K_STEP");
    auto &l2_x =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L2_X");
    auto &l2_y =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L2_Y");
    auto &l2_k_step =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L2_K_STEP");

    const double y_reg_value =
        y_base_width.get_raw_value() * native_vector_width;

    // register parameters are always correct, never changed

    l1_x.to_nearest_valid(x_reg.get_raw_value());
    l2_x.to_nearest_valid(l1_x.get_raw_value());

    l1_y.to_nearest_valid(y_reg_value);
    l2_y.to_nearest_valid(l1_y.get_raw_value());

    l2_k_step.to_nearest_valid(l1_k_step.get_raw_value());
  };

  // tune with parallel line search, large scenario
  {
    std::cout << "----------------- starting tuning with parallel line search, "
                 "large ------------ "
              << std::endl;
    std::uint64_t N_large = 4096;
    size_t repetitions = 5;
    combined::combined m(N_large, A_4096, B_4096, repetitions, verbose);

    size_t line_search_steps = 50;
    size_t restarts = 3;
    for (size_t restart = 0; restart < restarts; restart++) {
      std::cout << "restart: " << restart << std::endl;
      bool valid_start_found = false;
      while (!valid_start_found) {
        for (size_t parameter_index = 0; parameter_index < parameters.size();
             parameter_index++) {
          auto &p = parameters[parameter_index];
          p->set_random_value();
        }
        autotune::parameter_value_set parameter_values =
            autotune::to_parameter_values(parameters);
        if (precompile_validate_parameter_functor(parameter_values)) {
          valid_start_found = true;
        }
      }

      autotune::tuners::parallel_line_search tuner(
          autotune::combined_kernel, parameters, line_search_steps);
      tuner.set_parameter_adjustment_functor(parameter_adjustment_functor);
      tuner.set_verbose(true);
      tuner.set_write_measurement(scenario_name +
                                  "_parallel_line_search_large_" +
                                  std::to_string(restart));
      std::function<bool(const std::vector<double> &C)> test_result =
          [&C_reference_4096, N_large](const std::vector<double> &C) -> bool {
        for (size_t i = 0; i < N_large * N_large; i++) {
          double threshold = 1E-8;
          if (fabs(C[i] - C_reference_4096[i]) >= threshold) {
            std::cout << "test error C: " << C[i]
                      << " C_ref: " << C_reference_4096[i] << " i: " << i
                      << " (threshold: " << threshold << ")" << std::endl;
            return false;
          }
        }
        return true;
      };

      tuner.setup_test(test_result);

      autotune::countable_set optimal_parameters;
      {
        std::chrono::high_resolution_clock::time_point start =
            std::chrono::high_resolution_clock::now();
        optimal_parameters = tuner.tune(m.N_org, m.A_org, m.B_org,
                                        m.repetitions, duration_kernel, gflops_kernel);
        std::chrono::high_resolution_clock::time_point end =
            std::chrono::high_resolution_clock::now();
        double tuning_duration =
            std::chrono::duration<double>(end - start).count();
        tuner_duration_file << "parallel_line_search_large, " << tuning_duration
                            << std::endl;
      }

      std::cout << "----------------------- end tuning -----------------------"
                << std::endl;
      std::cout << "optimal parameter values (parallel line search, large):"
                << std::endl;
      optimal_parameters.print_values();
      autotune::combined_kernel.set_parameter_values(optimal_parameters);
      autotune::combined_kernel.compile();

      std::vector<double> C = m.matrix_multiply(duration_kernel, gflops_kernel);
      bool test_ok = test_result(C);
      if (test_ok) {
        std::cout << "optimal parameters test ok!" << std::endl;
      } else {
        std::cout << "optimal parameters FAILED test!" << std::endl;
      }

      double flops = 2 * static_cast<double>(N_large) *
                     static_cast<double>(N_large) *
                     static_cast<double>(N_large);
      double gflop = flops / 1E9;
      std::cout << "optimal inner_duration (parallel line search, large): "
                << duration_kernel << std::endl;
      std::cout << "[N = " << N_large << "] performance: "
                << ((repetitions * gflop) / duration_kernel) << "GFLOPS"
                << std::endl;
    }
    for (size_t parameter_index = 0; parameter_index < parameters.size();
         parameter_index++) {
      auto &p = parameters[parameter_index];
      p->set_initial();
    }
  }
  // tune with parallel line search, noisy scenario
  {
    std::cout << "----------------- starting tuning with parallel line search, "
                 "noisy ------------ "
              << std::endl;
    std::uint64_t N_noisy = 4096;
    size_t repetitions = 1;
    combined::combined m(N_noisy, A_4096, B_4096, repetitions,
                         verbose); // different number of repetitions

    size_t line_search_steps = 50;
    size_t restarts = 3;
    for (size_t restart = 0; restart < restarts; restart++) {
      std::cout << "restart: " << restart << std::endl;
      bool valid_start_found = false;
      while (!valid_start_found) {
        for (size_t parameter_index = 0; parameter_index < parameters.size();
             parameter_index++) {
          auto &p = parameters[parameter_index];
          p->set_random_value();
        }
        autotune::parameter_value_set parameter_values =
            autotune::to_parameter_values(parameters);
        if (precompile_validate_parameter_functor(parameter_values)) {
          valid_start_found = true;
        }
      }

      autotune::tuners::parallel_line_search tuner(
          autotune::combined_kernel, parameters, line_search_steps);
      tuner.set_parameter_adjustment_functor(parameter_adjustment_functor);
      tuner.set_verbose(true);
      tuner.set_write_measurement(scenario_name +
                                  "_parallel_line_search_noisy_" +
                                  std::to_string(restart));
      std::function<bool(const std::vector<double> &C)> test_result =
          [&C_reference_4096, N_noisy](const std::vector<double> &C) -> bool {
        for (size_t i = 0; i < N_noisy * N_noisy; i++) {
          double threshold = 1E-8;
          if (fabs(C[i] - C_reference_4096[i]) >= threshold) {
            std::cout << "test error C: " << C[i]
                      << " C_ref: " << C_reference_4096[i] << " i: " << i
                      << " (threshold: " << threshold << ")" << std::endl;
            return false;
          }
        }
        return true;
      };
      tuner.setup_test(test_result);

      autotune::countable_set optimal_parameters;
      {
        std::chrono::high_resolution_clock::time_point start =
            std::chrono::high_resolution_clock::now();
        optimal_parameters = tuner.tune(m.N_org, m.A_org, m.B_org,
                                        m.repetitions, duration_kernel, gflops_kernel);
        std::chrono::high_resolution_clock::time_point end =
            std::chrono::high_resolution_clock::now();
        double tuning_duration =
            std::chrono::duration<double>(end - start).count();
        tuner_duration_file << "parallel_line_search_noisy, " << tuning_duration
                            << std::endl;
      }

      std::cout << "----------------------- end tuning -----------------------"
                << std::endl;
      std::cout << "optimal parameter values (parallel line search, noisy):"
                << std::endl;
      optimal_parameters.print_values();
      autotune::combined_kernel.set_parameter_values(optimal_parameters);
      autotune::combined_kernel.compile();

      std::vector<double> C = m.matrix_multiply(duration_kernel, gflops_kernel);
      bool test_ok = test_result(C);
      if (test_ok) {
        std::cout << "optimal parameters test ok!" << std::endl;
      } else {
        std::cout << "optimal parameters FAILED test!" << std::endl;
      }

      double flops = 2 * static_cast<double>(N_noisy) *
                     static_cast<double>(N_noisy) *
                     static_cast<double>(N_noisy);
      double gflop = flops / 1E9;
      std::cout << "optimal inner_duration (parallel line search, noisy): "
                << duration_kernel << std::endl;
      std::cout << "[N = " << N_noisy << "] performance: "
                << ((repetitions * gflop) / duration_kernel) << "GFLOPS"
                << std::endl;
    }
    for (size_t parameter_index = 0; parameter_index < parameters.size();
         parameter_index++) {
      auto &p = parameters[parameter_index];
      p->set_initial();
    }
  }
}
