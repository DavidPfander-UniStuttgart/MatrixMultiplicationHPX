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

namespace detail {
std::uint64_t N = 4096;  // cmd argument
size_t repetitions = 10; // cmd argument
size_t repetitions_pvn_compare = 10;
size_t restarts = 1;
bool use_pvn = true;
std::vector<double> A;
std::vector<double> B;
std::vector<double> C_reference;
double duration_kernel;
double gflops_kernel;
std::vector<size_t> thread_values;
size_t native_vector_width;
std::map<std::string, std::string> pvn_values_map;

std::function<bool(const std::vector<double> &C)> test_result =
    [&](const std::vector<double> &C) -> bool {
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
} // namespace detail

template <typename parameter_set_type, typename F>
void evaluate_pvn(parameter_set_type &parameters, std::ofstream &pvn_csv_file,
                  const std::string &key, F &parameter_values_adjust_functor) {
  std::cout << "evaluate_pvn: " << key << std::endl;
  std::string old_value;
  if (key.compare("None") != 0) {
    old_value = parameters.get_by_name(key)->get_value();
    parameters.get_by_name(key)->set_value_unsafe(detail::pvn_values_map[key]);
  }
  autotune::parameter_value_set adjusted = to_parameter_values(parameters);
  parameter_values_adjust_functor(adjusted);
  autotune::combined_kernel.set_parameter_values(adjusted);
  std::vector<double> C = autotune::combined_kernel(
      detail::N, detail::A, detail::B, detail::repetitions_pvn_compare,
      detail::duration_kernel, detail::gflops_kernel);
  bool test_ok = detail::test_result(C);
  if (test_ok) {
    std::cout << "optimal parameters test ok!" << std::endl;
  } else {
    std::cout << "optimal parameters FAILED test!" << std::endl;
  }
  for (size_t i = 0; i < parameters.size(); i += 1) {
    pvn_csv_file << parameters[i]->get_value() << ",";
  }
  pvn_csv_file << key << "," << detail::duration_kernel << ","
               << detail::gflops_kernel << std::endl;
  if (key.compare("None") != 0) {
    parameters.get_by_name(key)->set_value_unsafe(old_value);
  }
}

template <typename parameter_set_type, typename F>
void evaluate_pvn_group(parameter_set_type &parameters,
                        std::ofstream &pvn_csv_file,
                        const std::string &group_name,
                        const std::vector<std::string> &keys,
                        F &parameter_values_adjust_functor) {
  std::cout << "evaluate_pvn_group: ";
  for (const std::string &key : keys) {
    std::cout << key << " ";
  }
  std::cout << std::endl;
  std::vector<std::string> old_values;
  if (keys.size() > 0) {
    for (const std::string &key : keys) {
      old_values.push_back(parameters.get_by_name(key)->get_value());
    }
    for (const std::string &key : keys) {
      parameters.get_by_name(key)->set_value_unsafe(
          detail::pvn_values_map[key]);
    }
  }
  autotune::parameter_value_set adjusted = to_parameter_values(parameters);
  parameter_values_adjust_functor(adjusted);
  autotune::combined_kernel.set_parameter_values(adjusted);
  std::vector<double> C = autotune::combined_kernel(
      detail::N, detail::A, detail::B, detail::repetitions_pvn_compare,
      detail::duration_kernel, detail::gflops_kernel);
  bool test_ok = detail::test_result(C);
  if (test_ok) {
    std::cout << "optimal parameters test ok!" << std::endl;
  } else {
    std::cout << "optimal parameters FAILED test!" << std::endl;
  }
  for (size_t i = 0; i < parameters.size(); i += 1) {
    pvn_csv_file << parameters[i]->get_value() << ",";
  }
  pvn_csv_file << group_name << "," << detail::duration_kernel << ","
               << detail::gflops_kernel << std::endl;
  if (keys.size() > 0) {
    for (size_t i = 0; i < keys.size(); i += 1) {
      parameters.get_by_name(keys[i])->set_value_unsafe(old_values[i]);
    }
  }
}

template <typename parameter_set_type, typename F>
void pvn_compare(const std::string &scenario_name,
                 parameter_set_type &parameters,
                 F &parameter_values_adjust_functor) {
  // disable L3 by setting them to value of L2 each
  // disable L2 by setting them to value of L1 each
  // disable L1 by setting them to min
  // disable X_REG by setting it to 1
  // disable Y_BASE_WIDTH by setting it to 1
  std::cout << "starting pvn_compare..." << std::endl;

  std::ofstream pvn_csv_file(scenario_name + "_pvn_compare.csv");

  for (size_t i = 0; i < parameters.size(); i += 1) {
    pvn_csv_file << parameters[i]->get_name() << ",";
  }
  pvn_csv_file << "changed_key,duration,gflops" << std::endl;

  evaluate_pvn(parameters, pvn_csv_file, "None",
               parameter_values_adjust_functor);
  evaluate_pvn(parameters, pvn_csv_file, "KERNEL_NUMA",
               parameter_values_adjust_functor);
  evaluate_pvn(parameters, pvn_csv_file, "KERNEL_SCHEDULE",
               parameter_values_adjust_functor);
  // register parameters
  evaluate_pvn(parameters, pvn_csv_file, "X_REG",
               parameter_values_adjust_functor);
  evaluate_pvn(parameters, pvn_csv_file, "Y_BASE_WIDTH",
               parameter_values_adjust_functor);
  // L1 parameters
  evaluate_pvn(parameters, pvn_csv_file, "L1_X",
               parameter_values_adjust_functor);
  evaluate_pvn(parameters, pvn_csv_file, "L1_Y",
               parameter_values_adjust_functor);
  evaluate_pvn(parameters, pvn_csv_file, "L1_K",
               parameter_values_adjust_functor);
  evaluate_pvn_group(parameters, pvn_csv_file, "L1_GROUP",
                     {"L1_X", "L1_Y", "L1_K"}, parameter_values_adjust_functor);
  // L3 parameters
  evaluate_pvn(parameters, pvn_csv_file, "L3_X",
               parameter_values_adjust_functor);
  evaluate_pvn(parameters, pvn_csv_file, "L3_Y",
               parameter_values_adjust_functor);
  evaluate_pvn(parameters, pvn_csv_file, "L3_K",
               parameter_values_adjust_functor);

  evaluate_pvn_group(parameters, pvn_csv_file, "REG_GROUP",
                     {"X_REG", "Y_BASE_WIDTH"},
                     parameter_values_adjust_functor);

  evaluate_pvn_group(parameters, pvn_csv_file, "L3_GROUP",
                     {"L3_X", "L3_Y", "L3_K"}, parameter_values_adjust_functor);
  // other parameters
  evaluate_pvn(parameters, pvn_csv_file, "KERNEL_OMP_THREADS",
               parameter_values_adjust_functor);
}

template <typename parameter_set_type, typename tuner_t, typename F>
void do_tuning(tuner_t &tuner, parameter_set_type &ps,
               const std::string &scenario_name, const std::string &tuner_name,
               F &parameter_values_adjust_functor) {
  std::cout << "----------------- starting tuning, scenario name: "
            << scenario_name << " ------------ " << std::endl;
  for (size_t restart = 0; restart < detail::restarts; restart++) {
    std::cout << "restart: " << restart << std::endl;
    if (!detail::use_pvn) {
      bool valid_start_found = false;
      while (!valid_start_found) {
        for (size_t parameter_index = 0; parameter_index < ps.size();
             parameter_index++) {
          auto &p = ps[parameter_index];
          p->set_random_value();
        }
        valid_start_found = true; // should get adjusted
      }
    } else {
      for (auto &pair : detail::pvn_values_map) {
        ps.get_by_name(pair.first)->set_value_unsafe(pair.second);
      }
    }

    tuner.set_verbose(true);
    tuner.set_write_measurement(scenario_name + std::string("_") +
                                std::to_string(restart));
    tuner.setup_test(detail::test_result);

    parameter_set_type optimal_parameters;
    {
      std::chrono::high_resolution_clock::time_point start =
          std::chrono::high_resolution_clock::now();
      optimal_parameters =
          tuner.tune(detail::N, detail::A, detail::B, detail::repetitions,
                     detail::duration_kernel, detail::gflops_kernel);
      std::chrono::high_resolution_clock::time_point end =
          std::chrono::high_resolution_clock::now();
      double tuning_duration =
          std::chrono::duration<double>(end - start).count();
      tuner_duration_file << tuner_name << ", " << tuning_duration << std::endl;
    }
    autotune::parameter_value_set pv =
        autotune::to_parameter_values(optimal_parameters);
    autotune::parameter_values_to_file(pv, scenario_name + std::string("_") +
                                               std::to_string(restart) +
                                               std::string(".json"));

    std::cout << "----------------------- end tuning -----------------------"
              << std::endl;

    pvn_compare(scenario_name + std::string("_") + std::to_string(restart),
                optimal_parameters, parameter_values_adjust_functor);

    std::cout
        << "----------------------- end pvn compare -----------------------"
        << std::endl;
    std::cout << "optimal parameter values:" << std::endl;
    optimal_parameters.print_values();
    // autotune::combined_kernel.set_parameter_values(optimal_parameters);
    // autotune::combined_kernel.compile();

    // std::vector<double> C = autotune::combined_kernel(
    //     detail::N, detail::A, detail::B, detail::repetitions,
    //     detail::duration_kernel, detail::gflops_kernel);
    // bool test_ok = detail::test_result(C);
    // if (test_ok) {
    //   std::cout << "optimal parameters test ok!" << std::endl;
    // } else {
    //   std::cout << "optimal parameters FAILED test!" << std::endl;
    // }
  }
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Error: not enough arguments!" << std::endl;
    return 1;
  } else if (argc > 5) {
    std::cerr << "Error: two many arguments given!" << std::endl;
    std::cerr << "args: node name; scenario suffix; N; rep" << std::endl;
    return 1;
  }
  std::string scenario_name(argv[1]);
  scenario_name += std::string("_") + std::string(argv[2]);
  std::string node_name(argv[1]);
  std::cout << "scenario_name: " << scenario_name << std::endl;
  detail::N = stod(std::string(argv[3]));
  std::cout << "N: " << detail::N << std::endl;
  detail::repetitions = stod(std::string(argv[4]));
  std::cout << "repetitions: " << detail::repetitions << std::endl;
  scenario_name = scenario_name + std::string("_") + std::to_string(detail::N) +
                  std::string("_") + std::to_string(detail::repetitions) +
                  std::string("r");

  //   uint64_t l1_size_bytes = 0;
  //   uint64_t l2_size_bytes = 0;
  // #ifdef WITH_LIKWID
  //   // use likwid to query some hardware information
  //   {
  //     int err = topology_init();
  //     if (err < 0) {
  //       std::cerr << "Unable to initialize likwid" << std::endl;
  //       return 1;
  //     } else {
  //       std::cout << "info: using likwid to query hardware information"
  //                 << std::endl;
  //     }
  //     // CpuInfo_t contains global information like name, CPU family, ...
  //     CpuInfo_t info = get_cpuInfo();
  //     std::cout << "using cpu name: " << info->name << std::endl;
  //     // CpuTopology_t contains information about the topology of the CPUs.
  //     CpuTopology_t topo = get_cpuTopology();
  //     for (size_t i = 0; i < topo->numCacheLevels; i++) {
  //       std::cout << "level: " << topo->cacheLevels[i].level << std::endl;
  //       std::cout << "size: " << topo->cacheLevels[i].size << std::endl;
  //       if (topo->cacheLevels[i].level == 1) {
  //         l1_size_bytes = topo->cacheLevels[i].size;
  //       } else if (topo->cacheLevels[i].level == 2) {
  //         l2_size_bytes = topo->cacheLevels[i].size;
  //       }
  //     }
  //   }
  // #else
  //   std::cout << "Not using likwid, querying internal database for hardware
  //   specs"
  //             << std::endl;
  //   if (scenario_name_raw.compare("6700k") == 0) {
  //     l1_size_bytes = 32 * 1024;
  //     l2_size_bytes = 256 * 1024;
  //   } else if (scenario_name_raw.compare("4300U") == 0) {
  //     l1_size_bytes = 32 * 1024;
  //     l2_size_bytes = 256 * 1024;
  //   } else if (scenario_name_raw.compare("xeonsilver") == 0) {
  //     l1_size_bytes = 32 * 1024;
  //     l2_size_bytes = 1024 * 1024;
  //   } else if (scenario_name_raw.compare("xeongold") == 0) {
  //     l1_size_bytes = 32 * 1024;
  //     l2_size_bytes = 1024 * 1024;
  //   } else if (scenario_name_raw.compare("knl") == 0) {
  //     l1_size_bytes = 32 * 1024;
  //     l2_size_bytes = 512 * 1024;
  //   } else if (scenario_name_raw.compare("epyc") == 0) {
  //     l1_size_bytes = 32 * 1024;
  //     l2_size_bytes = 512 * 1024;
  //   } else if (scenario_name_raw.compare("large") == 0) {
  //     l1_size_bytes = 32 * 1024;
  //     l2_size_bytes = 256 * 1024;
  //   } else if (scenario_name_raw.compare("element") == 0) {
  //     l1_size_bytes = 32 * 1024;
  //     l2_size_bytes = 512 * 1024;
  //   } else if (scenario_name_raw.compare("A10") == 0) {
  //     l1_size_bytes = 16 * 1024;
  //     l2_size_bytes = 2048 * 1024;
  //   } else {
  //     std::cerr
  //         << "error: platform hardware unknown and not compiled with
  //         liblikwid, "
  //            "aborting..."
  //         << std::endl;
  //     return 1;
  //   }
  //   std::cout << "level: " << 1 << std::endl;
  //   std::cout << "size: " << l1_size_bytes << std::endl;
  //   std::cout << "level: " << 2 << std::endl;
  //   std::cout << "size: " << l2_size_bytes << std::endl;
  // #endif

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

  tuner_duration_file.open(scenario_name + "_tuner_duration.csv");
  tuner_duration_file << "tuner, duration" << std::endl;

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

  auto parameter_adjustment_functor =
      [](autotune::countable_set &parameters) -> void {
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
    auto &l1_k =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L1_K");
    // auto &l2_x =
    //     parameters.get_by_name<autotune::countable_continuous_parameter>(
    //         "L2_X");
    // auto &l2_y =
    //     parameters.get_by_name<autotune::countable_continuous_parameter>(
    //         "L2_Y");
    // auto &l2_k =
    //     parameters.get_by_name<autotune::countable_continuous_parameter>(
    //         "L2_K");
    auto &l3_x =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L3_X");
    auto &l3_y =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L3_Y");
    auto &l3_k =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L3_K");

    const double y_reg_value =
        y_base_width.get_raw_value() * detail::native_vector_width;

    // register parameters are always correct, never changed
    l1_x.to_nearest_valid(x_reg.get_raw_value());
    // l2_x.to_nearest_valid(l1_x.get_raw_value());
    l3_x.to_nearest_valid(l1_x.get_raw_value());

    l1_y.to_nearest_valid(y_reg_value);
    // l2_y.to_nearest_valid(l1_y.get_raw_value());
    l3_y.to_nearest_valid(l1_y.get_raw_value());

    // l2_k.to_nearest_valid(l1_k.get_raw_value());
    l3_k.to_nearest_valid(l1_k.get_raw_value());
  };

  auto parameter_values_adjust_functor =
      [&](autotune::parameter_value_set &parameter_values) -> void {
    double X_REG = stod(parameter_values["X_REG"]);
    double Y_BASE_WIDTH = stod(parameter_values["Y_BASE_WIDTH"]);
    double L1_X = stod(parameter_values["L1_X"]);
    double L1_Y = stod(parameter_values["L1_Y"]);
    double L1_K = stod(parameter_values["L1_K"]);
    // double L2_X = stod(parameter_values["L2_X"]);
    // double L2_Y = stod(parameter_values["L2_Y"]);
    // double L2_K = stod(parameter_values["L2_K"]);
    double L3_X = stod(parameter_values["L3_X"]);
    double L3_Y = stod(parameter_values["L3_Y"]);
    double L3_K = stod(parameter_values["L3_K"]);

    const double Y_REG =
        Y_BASE_WIDTH * static_cast<double>(detail::native_vector_width);

    // register parameters are always correct, never changed
    // autotune::detail::round_to_nearest_bounded(current, factor, min, max);
    L1_X = autotune::detail::round_to_nearest_bounded(
        L1_X, X_REG, p_l1x.get_min(), p_l1x.get_max());
    // L2_X = autotune::detail::round_to_nearest_bounded(L2_X, L1_X,
    // p6.get_min(),
    //                                                   p6.get_max());
    L3_X = autotune::detail::round_to_nearest_bounded(
        L3_X, L1_X, p_l3x.get_min(), p_l3x.get_max());

    L1_Y = autotune::detail::round_to_nearest_bounded(
        L1_Y, Y_REG, p_l1y.get_min(), p_l1y.get_max());
    // L2_Y = autotune::detail::round_to_nearest_bounded(L2_Y, L1_Y,
    // p7.get_min(),
    //                                                   p7.get_max());
    L3_Y = autotune::detail::round_to_nearest_bounded(
        L3_Y, L1_Y, p_l3y.get_min(), p_l3y.get_max());

    // L2_K = autotune::detail::round_to_nearest_bounded(L2_K, L1_K,
    // p8.get_min(),
    //                                                   p8.get_max());
    L3_K = autotune::detail::round_to_nearest_bounded(
        L3_K, L1_K, p_l3k.get_min(), p_l3k.get_max());

    parameter_values["L1_X"] = autotune::detail::truncate_trailing_zeros(L1_X);
    // parameter_values["L2_X"] =
    // autotune::detail::truncate_trailing_zeros(L2_X);
    parameter_values["L3_X"] = autotune::detail::truncate_trailing_zeros(L3_X);

    parameter_values["L1_Y"] = autotune::detail::truncate_trailing_zeros(L1_Y);
    // parameter_values["L2_Y"] =
    // autotune::detail::truncate_trailing_zeros(L2_Y);
    parameter_values["L3_Y"] = autotune::detail::truncate_trailing_zeros(L3_Y);

    // parameter_values["L2_K"] =
    // autotune::detail::truncate_trailing_zeros(L2_K);
    parameter_values["L3_K"] = autotune::detail::truncate_trailing_zeros(L3_K);
  };

#ifdef DO_MONTE_CARLO
  auto parameter_adjustment_functor_randomizable =
      [](autotune::randomizable_set &parameters) -> void {
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
            "L1_K");
    // auto &l2_x =
    //     parameters.get_by_name<autotune::countable_continuous_parameter>(
    //         "L2_X");
    // auto &l2_y =
    //     parameters.get_by_name<autotune::countable_continuous_parameter>(
    //         "L2_Y");
    // auto &l2_k_step =
    //     parameters.get_by_name<autotune::countable_continuous_parameter>(
    //         "L2_K");
    auto &l3_x =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L3_X");
    auto &l3_y =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L3_Y");
    auto &l3_k_step =
        parameters.get_by_name<autotune::countable_continuous_parameter>(
            "L3_K");

    const double y_reg_value =
        y_base_width.get_raw_value() * detail::native_vector_width;

    // register parameters are always correct, never changed

    l1_x.to_nearest_valid(x_reg.get_raw_value());
    // l2_x.to_nearest_valid(l1_x.get_raw_value());
    l3_x.to_nearest_valid(l1_x.get_raw_value());

    l1_y.to_nearest_valid(y_reg_value);
    // l2_y.to_nearest_valid(l1_y.get_raw_value());
    l3_y.to_nearest_valid(l1_y.get_raw_value());

    // l2_k_step.to_nearest_valid(l1_k_step.get_raw_value());
    l3_k_step.to_nearest_valid(l1_k_step.get_raw_value());
  };
#endif

  detail::pvn_values_map.emplace("KERNEL_NUMA", "0");
  detail::pvn_values_map.emplace("KERNEL_SCHEDULE", "0");
  detail::pvn_values_map.emplace("X_REG", "1");
  detail::pvn_values_map.emplace("Y_BASE_WIDTH", "1");
  detail::pvn_values_map.emplace("L1_X", "8");
  detail::pvn_values_map.emplace("L1_Y", "8");
  detail::pvn_values_map.emplace("L1_K", "8");
  // detail::pvn_values_map.emplace("L2_X", "1");
  // detail::pvn_values_map.emplace("L2_Y", "16");
  // detail::pvn_values_map.emplace("L2_K", "1");
  detail::pvn_values_map.emplace("L3_X", "64");
  detail::pvn_values_map.emplace("L3_Y", "64");
  detail::pvn_values_map.emplace("L3_K", "64");
  std::string first_thread_value = std::to_string(detail::thread_values[0]);
  detail::pvn_values_map.emplace("KERNEL_OMP_THREADS",
                                 first_thread_value.c_str());

  // parameters.get_by_name("KERNEL_NUMA")->set_value_unsafe("1");
  // parameters.get_by_name("KERNEL_SCHEDULE")->set_value_unsafe("1");
  // parameters.get_by_name("X_REG")->set_value_unsafe("5");
  // parameters.get_by_name("Y_BASE_WIDTH")->set_value_unsafe("4");
  // parameters.get_by_name("L1_X")->set_value_unsafe("25");
  // parameters.get_by_name("L1_Y")->set_value_unsafe("32");
  // parameters.get_by_name("L1_K")->set_value_unsafe("64");
  // parameters.get_by_name("L2_X")->set_value_unsafe("100");
  // parameters.get_by_name("L2_Y")->set_value_unsafe("128");
  // parameters.get_by_name("L2_K")->set_value_unsafe("64");
  // parameters.get_by_name("L3_X")->set_value_unsafe("100");
  // parameters.get_by_name("L3_Y")->set_value_unsafe("128");
  // parameters.get_by_name("L3_K")->set_value_unsafe("256");
  // parameters.get_by_name("KERNEL_OMP_THREADS")
  //     ->set_value_unsafe(std::to_string(detail::thread_values[0]));
  // pvn_compare(scenario_name + std::string("_") + std::to_string(0),
  // parameters); return 0;

#ifdef DO_LINE_SEARCH
  {
    std::cout << "----------------- starting tuning with line search "
                 "-----------------"
              << std::endl;
    size_t line_search_steps = 50;
    autotune::tuners::line_search tuner(autotune::combined_kernel, parameters,
                                        line_search_steps);
    tuner.set_parameter_adjustment_functor(parameter_adjustment_functor);
    do_tuning(tuner, parameters, scenario_name + "_line_search", "line_search",
              parameter_values_adjust_functor);
  }
#endif
#ifdef DO_PARALLEL_LINE_SEARCH
  {
    std::cout << "----------------- starting tuning with parallel line search "
                 "-----------------"
              << std::endl;
    size_t line_search_steps = 50;
    autotune::tuners::parallel_line_search tuner(autotune::combined_kernel,
                                                 parameters, line_search_steps);
    tuner.set_parameter_adjustment_functor(parameter_adjustment_functor);
    do_tuning(tuner, parameters, scenario_name + "_parallel_line_search",
              "parallel_line_search", parameter_values_adjust_functor);
  }
#endif
#ifdef DO_NEIGHBOR_SEARCH
  {
    std::cout << "----------------- starting tuning with neighborhood search "
                 "----------------- "
              << std::endl;
    size_t search_steps = 50;
    autotune::tuners::neighborhood_search tuner(autotune::combined_kernel,
                                                parameters, search_steps);
    tuner.set_parameter_adjustment_functor(parameter_adjustment_functor);
    do_tuning(tuner, parameters, scenario_name + "_neighborhood_search",
              "neighborhood_search", parameter_values_adjust_functor);
  }
#endif
#ifdef DO_PARALLEL_NEIGHBOR_SEARCH
  {
    std::cout << "----------------- starting tuning with neighborhood search "
                 "----------------- "
              << std::endl;
    size_t search_steps = 50;
    autotune::tuners::parallel_neighborhood_search tuner(
        autotune::combined_kernel, parameters, search_steps);
    tuner.set_parameter_adjustment_functor(parameter_adjustment_functor);
    do_tuning(tuner, parameters,
              scenario_name + "_parallel_neighborhood_search",
              "parallel_neighborhood_search", parameter_values_adjust_functor);
  }
#endif
#ifdef DO_FULL_NEIGHBOR_SEARCH
  {
    std::cout
        << "----------------- starting tuning with full neighborhood search "
           "----------------- "
        << std::endl;
    size_t search_steps = 50;
    autotune::tuners::full_neighborhood_search tuner(autotune::combined_kernel,
                                                     parameters, search_steps);
    tuner.set_parameter_adjustment_functor(parameter_adjustment_functor);
    do_tuning(tuner, parameters, scenario_name + "_full_neighborhood_search",
              "full_neighborhood_search", parameter_values_adjust_functor);
  }
#endif
#ifdef DO_MONTE_CARLO
  {
    std::cout << "----------------- starting tuning with Monte Carlo search "
                 "-----------------"
              << std::endl;
    size_t search_steps = 50;
    autotune::tuners::monte_carlo tuner(autotune::combined_kernel,
                                        randomizable_parameters, search_steps,
                                        1000000);
    tuner.set_parameter_adjustment_functor(
        parameter_adjustment_functor_randomizable);
    do_tuning(tuner, randomizable_parameters, scenario_name + "_monte_carlo",
              "monte_carlo", parameter_values_adjust_functor);
  }
#endif
#ifdef DO_GREEDY_NEIGHBOR_SEARCH
  {
    std::cout
        << "----------------- starting tuning with greedy neighborhood search "
           "----------------- "
        << std::endl;
    size_t search_steps = 50;
    size_t changes_per_step = 2;
    autotune::tuners::greedy_neighborhood_search tuner(
        autotune::combined_kernel, detail::parameters, search_steps,
        changes_per_step);
    tuner.set_parameter_adjustment_functor(parameter_adjustment_functor);
    do_tuning(tuner, parameters, scenario_name + "_greedy_neighborhood_search",
              "greedy_neighborhood_search", parameter_values_adjust_functor);
  }
#endif
#ifdef DO_BRUTEFORCE
  {
    std::cout << "----------------- starting tuning with bruteforce "
                 "----------------- "
              << std::endl;
    autotune::tuners::bruteforce tuner(autotune::combined_kernel,
                                       detail::parameters);
    tuner.set_parameter_adjustment_functor(parameter_adjustment_functor);
    do_tuning<autotune::countable_set>(tuner, scenario_name + "bruteforce",
                                       "bruteforce",
                                       parameter_values_adjust_functor);
  }
#endif
#ifdef DO_PARALLEL_LINE_SEARCH_SPLIT
  {
    std::cout << "------ starting tuning with split line search "
                 "------"
              << std::endl;
    size_t line_search_steps = 10;
    size_t group_repeat = 3;

    autotune::parameter_value_set original_parameters =
        autotune::combined_kernel.get_parameter_values();

    autotune::parameter_value_set parameter_values;
    for (auto &pair : detail::pvn_values_map) {
      parameter_values[pair.first] = pair.second;
    }
    for (const std::string &name : {"L1_X", "L1_Y", "L1_K"}) {
      parameters_group_l1.get_by_name(name)->set_value_unsafe(
          detail::pvn_values_map[name]);
    }
    for (const std::string &name : {"L3_X", "L3_Y", "L3_K"}) {
      parameters_group_l3.get_by_name(name)->set_value_unsafe(
          detail::pvn_values_map[name]);
    }
    for (const std::string &name : {"X_REG", "Y_BASE_WIDTH"}) {
      parameters_group_register.get_by_name(name)->set_value_unsafe(
          detail::pvn_values_map[name]);
    }
    for (const std::string &name :
         {"KERNEL_OMP_THREADS", "KERNEL_SCHEDULE", "KERNEL_NUMA"}) {
      parameters_group_other.get_by_name(name)->set_value_unsafe(
          detail::pvn_values_map[name]);
    }

    // make sure that all parameters are known to the kernel
    autotune::combined_kernel.set_parameter_values(parameter_values);
    autotune::print_parameter_values(
        autotune::combined_kernel.get_parameter_values());

    // create all tuners
    autotune::tuners::parallel_line_search tuner_l3(
        autotune::combined_kernel, parameters_group_l3, line_search_steps);
    tuner_l3.set_parameter_values_adjustment_functor(
        parameter_values_adjust_functor);
    tuner_l3.set_write_measurement(scenario_name +
                                   "_split_parallel_line_search_l3");
    tuner_l3.setup_test(detail::test_result);

    // autotune::tuners::parallel_line_search tuner_l2(
    //     autotune::combined_kernel, parameters_group_l2, line_search_steps);
    // tuner_l2.set_parameter_values_adjustment_functor(
    //     parameter_values_adjust_functor);
    // tuner_l2.set_write_measurement(scenario_name +
    //                                "_split_parallel_line_search_l2");
    // tuner_l2.setup_test(detail::test_result);

    autotune::tuners::parallel_line_search tuner_l1(
        autotune::combined_kernel, parameters_group_l1, line_search_steps);
    tuner_l1.set_parameter_values_adjustment_functor(
        parameter_values_adjust_functor);
    tuner_l1.set_write_measurement(scenario_name +
                                   "_split_parallel_line_search_l1");
    tuner_l1.setup_test(detail::test_result);

    autotune::tuners::parallel_line_search tuner_reg(autotune::combined_kernel,
                                                     parameters_group_register,
                                                     line_search_steps);
    tuner_reg.set_parameter_values_adjustment_functor(
        parameter_values_adjust_functor);
    tuner_reg.set_write_measurement(scenario_name +
                                    "_split_parallel_line_search_register");
    tuner_reg.setup_test(detail::test_result);

    autotune::tuners::parallel_line_search tuner_other(
        autotune::combined_kernel, parameters_group_other, line_search_steps);
    tuner_other.set_parameter_values_adjustment_functor(
        parameter_values_adjust_functor);
    tuner_other.set_write_measurement(scenario_name +
                                      "_split_parallel_line_search_other");
    tuner_other.setup_test(detail::test_result);

    autotune::tuners::group_tuner g(autotune::combined_kernel, group_repeat,
                                    tuner_reg, tuner_l1, tuner_l3, // tuner_l2,
                                    tuner_other);
    g.set_verbose(true);
    g.set_write_measurement(scenario_name + "_split_parallel_line_search_meta");
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    autotune::parameter_value_set optimal_parameter_values =
        g.tune(m.N_org, m.A_org, m.B_org, m.repetitions,
               detail::duration_kernel, detail::gflops_kernel);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    double tuning_duration = std::chrono::duration<double>(end - start).count();
    tuner_duration_file << "split_parallel_line_search, " << tuning_duration
                        << std::endl;
    parameter_values_adjust_functor(optimal_parameter_values);
    autotune::combined_kernel.set_parameter_values(optimal_parameter_values);
    autotune::parameter_values_to_file(
        optimal_parameter_values,
        scenario_name + "_split_parallel_line_search_" + std::string(".json"));

    std::cout << "----------------------- end tuning -----------------------"
              << std::endl;

    // autotune::parameter_value_set optimal_parameter_values;
    // for (auto &pair : detail::pvn_values_map) {
    //   optimal_parameter_values[pair.first] = pair.second;
    // }
    // autotune::parameter_value_set optimal_parameter_values =
    //     autotune::parameter_values_from_file("large_test_4096_1r.json");
    // parameter_values_adjust_functor(optimal_parameter_values);
    // std::cout << "optimal_parameter_values:" << std::endl;
    // autotune::print_parameter_values(optimal_parameter_values);

    for (size_t i = 0; i < parameters.size(); i += 1) {
      parameters[i]->set_value_unsafe(
          optimal_parameter_values[parameters[i]->get_name()]);
    }

    std::cout << "optimal_parameter_values:" << std::endl;
    parameters.print_values();

    pvn_compare(scenario_name + "_split_parallel_line_search", parameters,
                parameter_values_adjust_functor);

    std::cout
        << "----------------------- end pvn compare -----------------------"
        << std::endl;
    std::cout << "optimal parameter values:" << std::endl;
    autotune::print_parameter_values(optimal_parameter_values);
  }
#endif
#ifdef DO_PARALLEL_FULL_NEIGHBOR_SEARCH_SPLIT
  {
    std::cout << "------ starting tuning with split line search "
                 "------"
              << std::endl;
    size_t line_search_steps = 10;
    size_t group_repeat = 3;

    autotune::parameter_value_set original_parameters =
        autotune::combined_kernel.get_parameter_values();

    autotune::parameter_value_set parameter_values;
    for (auto &pair : detail::pvn_values_map) {
      parameter_values[pair.first] = pair.second;
    }
    for (const std::string &name : {"L1_X", "L1_Y", "L1_K"}) {
      parameters_group_l1.get_by_name(name)->set_value_unsafe(
          detail::pvn_values_map[name]);
    }
    for (const std::string &name : {"L3_X", "L3_Y", "L3_K"}) {
      parameters_group_l3.get_by_name(name)->set_value_unsafe(
          detail::pvn_values_map[name]);
    }
    for (const std::string &name : {"X_REG", "Y_BASE_WIDTH"}) {
      parameters_group_register.get_by_name(name)->set_value_unsafe(
          detail::pvn_values_map[name]);
    }
    for (const std::string &name :
         {"KERNEL_OMP_THREADS", "KERNEL_SCHEDULE", "KERNEL_NUMA"}) {
      parameters_group_other.get_by_name(name)->set_value_unsafe(
          detail::pvn_values_map[name]);
    }

    // make sure that all parameters are known to the kernel
    autotune::combined_kernel.set_parameter_values(parameter_values);
    autotune::print_parameter_values(
        autotune::combined_kernel.get_parameter_values());

    // create all tuners
    autotune::tuners::parallel_full_neighborhood_search tuner_l3(
        autotune::combined_kernel, parameters_group_l3, line_search_steps);
    tuner_l3.set_parameter_values_adjustment_functor(
        parameter_values_adjust_functor);
    tuner_l3.set_write_measurement(
        scenario_name + "_split_parallel_full_neighborhood_search_l3");
    tuner_l3.setup_test(detail::test_result);

    // autotune::tuners::parallel_full_neighborhood_search tuner_l2(
    //     autotune::combined_kernel, parameters_group_l2, line_search_steps);
    // tuner_l2.set_parameter_values_adjustment_functor(
    //     parameter_values_adjust_functor);
    // tuner_l2.set_write_measurement(
    //     scenario_name + "_split_parallel_full_neighborhood_search_l2");
    // tuner_l2.setup_test(detail::test_result);

    autotune::tuners::parallel_full_neighborhood_search tuner_l1(
        autotune::combined_kernel, parameters_group_l1, line_search_steps);
    tuner_l1.set_parameter_values_adjustment_functor(
        parameter_values_adjust_functor);
    tuner_l1.set_write_measurement(
        scenario_name + "_split_parallel_full_neighborhood_search_l1");
    tuner_l1.setup_test(detail::test_result);

    autotune::tuners::parallel_full_neighborhood_search tuner_reg(
        autotune::combined_kernel, parameters_group_register,
        line_search_steps);
    tuner_reg.set_parameter_values_adjustment_functor(
        parameter_values_adjust_functor);
    tuner_reg.set_write_measurement(
        scenario_name + "_split_parallel_full_neighborhood_search_register");
    tuner_reg.setup_test(detail::test_result);

    autotune::tuners::parallel_full_neighborhood_search tuner_other(
        autotune::combined_kernel, parameters_group_other, line_search_steps);
    tuner_other.set_parameter_values_adjustment_functor(
        parameter_values_adjust_functor);
    tuner_other.set_write_measurement(
        scenario_name + "_split_parallel_full_neighborhood_search_other");
    tuner_other.setup_test(detail::test_result);

    autotune::tuners::group_tuner g(autotune::combined_kernel, group_repeat,
                                    tuner_reg, tuner_l1, tuner_l3, // tuner_l2,
                                    tuner_other);
    g.set_verbose(true);
    g.set_write_measurement(scenario_name +
                            "_split_parallel_full_neighborhood_search_meta");
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    autotune::parameter_value_set optimal_parameter_values =
        g.tune(m.N_org, m.A_org, m.B_org, m.repetitions,
               detail::duration_kernel, detail::gflops_kernel);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    double tuning_duration = std::chrono::duration<double>(end - start).count();
    tuner_duration_file << "split_parallel_full_neighborhood_search, "
                        << tuning_duration << std::endl;
    parameter_values_adjust_functor(optimal_parameter_values);
    autotune::combined_kernel.set_parameter_values(optimal_parameter_values);

    autotune::parameter_values_to_file(
        optimal_parameter_values,
        scenario_name + "_split_parallel_full_neighborhood_search_" +
            std::string(".json"));

    std::cout << "----------------------- end tuning -----------------------"
              << std::endl;
    for (size_t i = 0; i < parameters.size(); i += 1) {
      parameters[i]->set_value_unsafe(
          optimal_parameter_values[parameters[i]->get_name()]);
    }
    pvn_compare(scenario_name + "_split_parallel_full_neighborhood_search",
                parameters, parameter_values_adjust_functor);

    std::cout
        << "----------------------- end pvn compare -----------------------"
        << std::endl;
    std::cout << "optimal parameter values:" << std::endl;
    autotune::print_parameter_values(optimal_parameter_values);
  }
#endif
}
