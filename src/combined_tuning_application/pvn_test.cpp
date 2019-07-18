#include "autotune/autotune.hpp"
#include "autotune/parameter.hpp"
#include "autotune/parameter_value_set.hpp"
#include "autotune/tuners/bruteforce.hpp"
#include "autotune/tuners/full_neighborhood_search.hpp"
#include "autotune/tuners/greedy_neighborhood_search.hpp"
#include "autotune/tuners/group_tuner.hpp"
#include "autotune/tuners/line_search.hpp"
#include "autotune/tuners/monte_carlo.hpp"
#include "autotune/tuners/neighborhood_search.hpp"
#include "autotune/tuners/parallel_full_neighborhood_search.hpp"
#include "autotune/tuners/parallel_line_search.hpp"
#include "autotune/tuners/parallel_neighborhood_search.hpp"

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
#include <stdlib.h>

// #define WITH_LIBLIKWID // controlled by cmake

// #define DO_LINE_SEARCH
#define DO_PARALLEL_LINE_SEARCH
// #define DO_NEIGHBOR_SEARCH
#define DO_PARALLEL_NEIGHBOR_SEARCH
// #define DO_FULL_NEIGHBOR_SEARCH
#define DO_MONTE_CARLO
// #define DO_GREEDY_NEIGHBOR_SEARCH

#define DO_PARALLEL_LINE_SEARCH_SPLIT
#define DO_PARALLEL_FULL_NEIGHBOR_SEARCH_SPLIT
// #define DO_NEIGHBOR_SEARCH_SPLIT
// #define DO_GREEDY_NEIGHBOR_SEARCH_SPLIT
// #define DO_BRUTEFORCE

AUTOTUNE_KERNEL(uint64_t(), hardware_query_kernel,
                "src/variants/hardware_query_kernel")

std::ofstream tuner_duration_file;

// hacky
#include "combined_tuner_common.hpp"

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Error: not enough arguments!" << std::endl;
    return 1;
  } else if (argc > 3) {
    std::cerr << "Error: two many arguments given!" << std::endl;
    std::cerr << "args: node_name; parameters file" << std::endl;
    return 1;
  }
  std::string node_name(argv[1]);
  std::string parameter_values_file_name(argv[2]);

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
  detail::native_vector_width = autotune::hardware_query_kernel();
  std::cout << "native_vector_width: " << detail::native_vector_width
            << std::endl;

  bool transposed = false;
  bool verbose = false;

  // create matrices A, B>
  detail::A = util::create_random_matrix<double>(detail::N);
  detail::B = util::create_random_matrix<double>(detail::N);

  {
    kernel_tiled::kernel_tiled m_tiled(detail::N, detail::A, detail::B,
                                       transposed, 1, verbose);
    std::cout << "calculating reference solution..." << std::flush;
    double duration_reference;
    detail::C_reference = m_tiled.matrix_multiply(duration_reference);
  }

  std::cout << " done" << std::endl << std::flush;

  if (transposed) {
    throw util::matrix_multiplication_exception(
        "algorithm \"combined\" doens't allow B to be transposed");
  }
  combined::combined m(detail::N, detail::A, detail::B, detail::repetitions,
                       verbose);

  autotune::combined_kernel.set_verbose(true);
  autotune::combined_kernel.set_kernel_duration_functor(
      [&]() { return detail::duration_kernel; });

  auto &builder = autotune::combined_kernel.get_builder<cppjit::builder::gcc>();
  // builder.set_verbose(true);

  builder.set_include_paths(
      "-IAutoTuneTMP/AutoTuneTMP_install/include -Isrc/variants/ "
      "-IAutoTuneTMP/Vc_install/include "
      "-IAutoTuneTMP/boost_install/include -IAutoTuneTMP/likwid/src/includes");
  builder.set_cpp_flags("-Wall -Wextra -std=c++17 -march=native -mtune=native "
                        "-O3 -g -ffast-math -fopenmp -fPIC -fno-gnu-unique");
  builder.set_library_paths("-LAutoTuneTMP/likwid");
  builder.set_link_flags("-shared -g -fno-gnu-unique");
  builder.set_libraries("-lnuma AutoTuneTMP/likwid/liblikwid.so");

  autotune::countable_continuous_parameter p_regx("X_REG", 1, 1, 1, 5);
  autotune::countable_continuous_parameter p_regybase("Y_BASE_WIDTH", 1, 1, 1,
                                                      5);

  autotune::countable_continuous_parameter p_l1x("L1_X", 8, 8, 8, 128, true);
  autotune::countable_continuous_parameter p_l1y("L1_Y", 8, 8, 8, 128, true);
  autotune::countable_continuous_parameter p_l1k("L1_K", 8, 8, 8, 128, true);

  autotune::countable_continuous_parameter p_l3x("L3_X", 64, 64, 64, 512, true);
  autotune::countable_continuous_parameter p_l3y("L3_Y", 64, 64, 64, 512, true);
  autotune::countable_continuous_parameter p_l3k("L3_K", 64, 64, 64, 512, true);

  autotune::fixed_set_parameter<int> p_numa("KERNEL_NUMA",
                                            {0, 1}); // 0 == none, 1 == copy
  autotune::fixed_set_parameter<int> p_schedule(
      "KERNEL_SCHEDULE", {0, 1}); // 0==static, 1==dynamic

  // autotune::countable_continuous_parameter p6("L2_X", 1, 32, 1, 128, true);
  // autotune::countable_continuous_parameter p7("L2_Y", 16, 16, 16, 128, true);
  // autotune::countable_continuous_parameter p8("L2_K", 1, 32, 1, 256, true);

  int64_t smt_factor = 1;
  if (node_name.compare("knl") == 0) {
    smt_factor = 2;
  }
  size_t openmp_threads = omp_get_max_threads();
  detail::thread_values.push_back(openmp_threads);
  std::cout << "KERNEL_OMP_THREADS values: " << openmp_threads;

  bool is_omp_num_threads_set = false;
  {
    char *omp_num_threads_var;
    omp_num_threads_var = getenv("OMP_NUM_THREADS");
    if (omp_num_threads_var != nullptr) {
      is_omp_num_threads_set = true;
    }
  }

  if (!is_omp_num_threads_set) {
    for (int64_t i = 0; i < smt_factor; i++) { // 4-way HT assumed max
      openmp_threads /= 2;
      std::cout << ", " << openmp_threads;
      detail::thread_values.push_back(openmp_threads);
    }
  }
  autotune::fixed_set_parameter<size_t> p_threads("KERNEL_OMP_THREADS",
                                                  detail::thread_values);
  std::cout << std::endl;

  autotune::countable_set parameters_group_register;
  autotune::countable_set parameters_group_l1;
  // autotune::countable_set parameters_group_l2;
  autotune::countable_set parameters_group_l3;
  autotune::countable_set parameters_group_other;

  // parameters_group_l2.add_parameter(p6);
  // parameters_group_l2.add_parameter(p7);
  // parameters_group_l2.add_parameter(p8);

  parameters_group_register.add_parameter(p_regx);
  parameters_group_register.add_parameter(p_regybase);
  parameters_group_l1.add_parameter(p_l1x);
  parameters_group_l1.add_parameter(p_l1y);
  parameters_group_l1.add_parameter(p_l1k);
  parameters_group_l3.add_parameter(p_l3x);
  parameters_group_l3.add_parameter(p_l3y);
  parameters_group_l3.add_parameter(p_l3k);
  parameters_group_other.add_parameter(p_numa);
  parameters_group_other.add_parameter(p_schedule);
  parameters_group_other.add_parameter(p_threads);

  autotune::countable_set parameters;
  parameters.add_parameter(p_regx);
  parameters.add_parameter(p_regybase);
  parameters.add_parameter(p_l1x);
  parameters.add_parameter(p_l1y);
  parameters.add_parameter(p_l1k);
  // parameters.add_parameter(p6);
  // parameters.add_parameter(p7);
  // parameters.add_parameter(p8);
  parameters.add_parameter(p_l3x);
  parameters.add_parameter(p_l3y);
  parameters.add_parameter(p_l3k);
  parameters.add_parameter(p_numa);
  parameters.add_parameter(p_schedule);
  parameters.add_parameter(p_threads);

  autotune::randomizable_set randomizable_parameters;
  randomizable_parameters.add_parameter(p_regx);
  randomizable_parameters.add_parameter(p_regybase);
  randomizable_parameters.add_parameter(p_l1x);
  randomizable_parameters.add_parameter(p_l1y);
  randomizable_parameters.add_parameter(p_l1k);
  // randomizable_parameters.add_parameter(p6);
  // randomizable_parameters.add_parameter(p7);
  // randomizable_parameters.add_parameter(p8);
  randomizable_parameters.add_parameter(p_l3x);
  randomizable_parameters.add_parameter(p_l3y);
  randomizable_parameters.add_parameter(p_l3k);
  randomizable_parameters.add_parameter(p_numa);
  randomizable_parameters.add_parameter(p_schedule);
  randomizable_parameters.add_parameter(p_threads);

  autotune::combined_kernel.set_source_dir("src/variants/combined_kernel");

  auto precompile_validate_parameter_functor =
      [](autotune::parameter_value_set &parameters) -> bool {
    int64_t X_REG = stol(parameters["X_REG"]);
    int64_t Y_BASE_WIDTH = stol(parameters["Y_BASE_WIDTH"]);
    int64_t L1_X = stol(parameters["L1_X"]);
    int64_t L1_Y = stol(parameters["L1_Y"]);
    int64_t L1_K = stol(parameters["L1_K"]);
    // int64_t L2_X = stol(parameters["L2_X"]);
    // int64_t L2_Y = stol(parameters["L2_Y"]);
    // int64_t L2_K = stol(parameters["L2_K"]);
    int64_t L3_X = stol(parameters["L3_X"]);
    int64_t L3_Y = stol(parameters["L3_Y"]);
    int64_t L3_K = stol(parameters["L3_K"]);

    const int64_t Y_REG = Y_BASE_WIDTH * detail::native_vector_width;

    if (L1_X < X_REG) {
      std::cout << "error: L1_X < X_REG, L1_X too small" << std::endl;
      return false;
    }
    if (L3_X < L1_X) {
      std::cout << "error: L3_X < L1_X, L3_X too small" << std::endl;
      return false;
    }
    if (L1_Y < Y_REG) {
      std::cout << "error: L1_Y < Y_REG, L1_Y too small" << std::endl;
      return false;
    }
    if (L3_Y < L1_Y) {
      std::cout << "error: L3_Y < L1_Y, L3_Y too small" << std::endl;
      return false;
    }
    if (L3_K < L1_K) {
      std::cout << "error: L3_K < L1_K, L3_K too small" << std::endl;
      return false;
    };
    if (L1_X % X_REG != 0) {
      std::cout << "error: L1_X does not divide X_REG" << std::endl;
      return false;
    }
    if (L1_Y % Y_REG != 0) {
      std::cout << "error: L1_Y does not divide Y_REG" << std::endl;
      return false;
    }
    // if (L2_X % L1_X != 0) {
    //   std::cout << "error: x direction blocking error: L2_X % L1_X != 0"
    //             << std::endl;
    //   return false;
    // }
    // if (L2_Y % L1_Y != 0) {
    //   std::cout << "error: y direction blocking error: L2_Y % L1_Y != 0"
    //             << std::endl;
    //   return false;
    // }
    // if (L2_K % L1_K != 0) {
    //   std::cout << "error: k direction blocking error: L2_K % L1_K != 0 "
    //             << std::endl;
    //   return false;
    // }
    if (L3_X % L1_X != 0) {
      std::cout << "error: x direction blocking error: L3_X % L1_X != 0"
                << std::endl;
      return false;
    }
    if (L3_Y % L1_Y != 0) {
      std::cout << "error: y direction blocking error: L3_Y % L1_Y != 0"
                << std::endl;
      return false;
    }
    if (L3_K % L1_K != 0) {
      std::cout << "error: k direction blocking error: L3_K % L1_K != 0 "
                << std::endl;
      return false;
    }

    // size_t l2_memory = (L2_X * L2_K + L2_K * L2_Y + L2_X * L2_Y) * 8;
    // if (l2_memory > l2_size_bytes) {
    //   std::cout << "rejected by l2 cache size requirement" << std::endl;
    //   return false;
    // }
    // size_t l1_memory = (L1_X * L1_K + L1_K * L1_Y + L1_X * L1_Y) * 8;
    // if (l1_memory > l1_size_bytes) {
    //   std::cout << "rejected by l1 cache size requirement" << std::endl;
    //   return false;
    // }

    return true;
  };
  autotune::combined_kernel.set_precompile_validate_parameter_functor(
      precompile_validate_parameter_functor);

  detail::pvn_values_map.emplace("KERNEL_NUMA", "0");
  detail::pvn_values_map.emplace("KERNEL_SCHEDULE", "0");
  detail::pvn_values_map.emplace("X_REG", "1");
  detail::pvn_values_map.emplace("Y_BASE_WIDTH", "1");
  detail::pvn_values_map.emplace("L1_X", "8");
  detail::pvn_values_map.emplace("L1_Y", "8");
  detail::pvn_values_map.emplace("L1_K", "8");
  detail::pvn_values_map.emplace("L3_X", "64");
  detail::pvn_values_map.emplace("L3_Y", "64");
  detail::pvn_values_map.emplace("L3_K", "64");
  std::string first_thread_value = std::to_string(detail::thread_values[0]);
  detail::pvn_values_map.emplace("KERNEL_OMP_THREADS",
                                 first_thread_value.c_str());

  autotune::parameter_value_set pv =
      autotune::parameter_values_from_file(parameter_values_file_name);
  for (size_t i = 0; i < parameters.size(); i += 1) {
    parameters[i]->set_value_unsafe(pv[parameters[i]->get_name()]);
  }
  pvn_compare(std::string("pvn_test"), parameters,
              parameter_values_adjust_functor);
}
