#include "combined.hpp"

#include <chrono>

#include "index_iterator.hpp"
#include "util/util.hpp"

#include "autotune/tuners/countable_set.hpp"

#include <Vc/Vc>
using Vc::double_v;

#include <omp.h>

AUTOTUNE_DEFINE_KERNEL(std::vector<double>(std::size_t, std::vector<double> &,
                                           std::vector<double> &, size_t,
                                           double &, double &),
                       combined_kernel, "src/variants/combined_kernel")

using namespace index_iterator;

namespace combined {

combined::combined(size_t N, std::vector<double> &A_org,
                   std::vector<double> &B_org, uint64_t repetitions,
                   uint64_t verbose)
    : N_org(N), A_org(A_org), B_org(B_org), repetitions(repetitions),
      verbose(verbose) {
  autotune::combined_kernel.set_verbose(verbose);
}

combined::~combined() {}

std::vector<double> combined::matrix_multiply(double &duration, double &gflops,
                                              bool set_default_parameters) {
  if (!autotune::combined_kernel.is_compiled()) {
    auto &builder =
        autotune::combined_kernel.get_builder<cppjit::builder::gcc>();
    builder.set_do_cleanup(false);
    // builder.set_verbose(true);
    builder.set_include_paths(
        "-IAutoTuneTMP/AutoTuneTMP_install/include -Isrc/variants/ "
        "-IAutoTuneTMP/Vc_install/include "
        "-IAutoTuneTMP/boost_install/include "
        "-IAutoTuneTMP/likwid/src/includes");
    builder.set_cpp_flags(
        "-Wall -Wextra -std=c++17 -march=native -mtune=native "
        "-O3 -g -ffast-math -fopenmp -fPIC -fno-gnu-unique");
    builder.set_link_flags("-shared -fno-gnu-unique");
    builder.set_library_paths("-LAutoTuneTMP/likwid");
    builder.set_libraries("-lnuma -llikwid");
    builder.set_builder_verbose(true);

    if (set_default_parameters) {
      autotune::countable_set parameters;
      autotune::fixed_set_parameter<int> p1("KERNEL_NUMA",
                                            {1}); // 0 == none, 1 == copy
      autotune::fixed_set_parameter<int> p2("KERNEL_SCHEDULE",
                                            {1}); // 0==static, 1==dynamic
      autotune::fixed_set_parameter<std::string> p3("L3_X", {"320"}, false);
      autotune::fixed_set_parameter<std::string> p4("L3_Y", {"384"}, false);
      autotune::fixed_set_parameter<std::string> p5("L3_K", {"100"}, false);
      // autotune::fixed_set_parameter<std::string> p6("L2_X", {"80"}, false);
      // autotune::fixed_set_parameter<std::string> p7("L2_Y", {"48"}, false);
      // autotune::fixed_set_parameter<std::string> p8("L2_K", {"100"}, false);
      autotune::fixed_set_parameter<std::string> p9("L1_X", {"80"}, false);
      autotune::fixed_set_parameter<std::string> p10("L1_Y", {"8"}, false);
      autotune::fixed_set_parameter<std::string> p11("L1_K", {"100"}, false);
      autotune::fixed_set_parameter<std::string> p12("X_REG", {"5"}, false);
      autotune::fixed_set_parameter<std::string> p13("Y_BASE_WIDTH", {"2"},
                                                     false);
      size_t openmp_threads = omp_get_max_threads();
      autotune::fixed_set_parameter<size_t> p14("KERNEL_OMP_THREADS",
                                                {openmp_threads});

      parameters.add_parameter(p1);
      parameters.add_parameter(p2);
      parameters.add_parameter(p3);
      parameters.add_parameter(p4);
      parameters.add_parameter(p5);
      // parameters.add_parameter(p6);
      // parameters.add_parameter(p7);
      // parameters.add_parameter(p8);
      parameters.add_parameter(p9);
      parameters.add_parameter(p10);
      parameters.add_parameter(p11);
      parameters.add_parameter(p12);
      parameters.add_parameter(p13);
      parameters.add_parameter(p14);

      autotune::combined_kernel.set_parameter_values(parameters);
    }
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
  C_return = autotune::combined_kernel(N_org, A_org, B_org, repetitions,
                                       duration, gflops);
  return C_return;
}
} // namespace combined
