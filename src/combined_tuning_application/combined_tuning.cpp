// #include <hpx/hpx_init.hpp>
// #include <hpx/include/actions.hpp>
// #include <hpx/include/async.hpp>
// #include <hpx/include/components.hpp>
// #include <hpx/include/util.hpp>

#include "autotune/autotune.hpp"
#include "autotune/parameter.hpp"

#include "reference_kernels/naive.hpp"
// #include "reference_kernels/kernel_test.hpp"
#include "reference_kernels/kernel_tiled.hpp"
#include "util/create_random_matrix.hpp"
#include "util/matrix_multiplication_exception.hpp"
#include "util/util.hpp"
#include "variants/combined.hpp"

#include <random>

std::uint64_t N = 4096;
bool transposed = false;
size_t repetitions = 5;
bool verbose = false;

// create matrices A, B>
std::vector<double> A;
std::vector<double> B;
std::vector<double> C_reference;

// int hpx_main() {

//   return hpx::finalize();
// }

int main(int argc, char **argv) {
  // create matrices A, B>
  A = util::create_random_matrix<double>(N);
  B = util::create_random_matrix<double>(N);

  std::cout << "calculating reference solution..." << std::flush;
  // kernel_test::kernel_test m_ref(N, A, B, transposed, repetitions, verbose);
  // C_reference = m_ref.matrix_multiply();
  kernel_tiled::kernel_tiled m_ref(N, A, B, transposed, repetitions, verbose);
  double duration_dummy;
  C_reference = m_ref.matrix_multiply(duration_dummy);
  // C_reference = naive_matrix_multiply(N, A, B);
  std::cout << " done" << std::endl << std::flush;

  // int return_value = hpx::init(argc, argv);

  if (transposed) {
    throw util::matrix_multiplication_exception(
        "algorithm \"combined\" doens't allow B to be transposed");
  }
  combined::combined m(N, A, B, repetitions, verbose);

  autotune::combined_kernel.set_verbose(true);

  auto builder =
      autotune::combined_kernel.get_builder_as<cppjit::builder::gcc>();
  builder->set_verbose(true);
  // builder->set_include_paths(
  //     "-I /home/winter/git/AutoTuneTMP/src -I src/variants/ -I "
  //     "/home/winter/hpx_install_with_symbols/include -I "
  //     "/home/winter/hpx_install_with_symbols/include/hpx/external -DNDEBUG "
  //     "-Wall -Wextra -std=c++14 -march=native -mtune=native -O3 -ffast-math "
  //     "-DHPX_APPLICATION_EXPORTS -fopenmp "
  //     "-DHPX_ENABLE_ASSERT_HANDLER -I/home/winter/Vc_head_install/include "
  //     "-I/home/winter/boost_1_63_0_install/include");

  builder->set_include_paths(
      "-I /home/winter/git/AutoTuneTMP/src -I src/variants/ -I "
      "-Wall -Wextra -std=c++14 -march=native -mtune=native -O3 -ffast-math "
      " -fopenmp "
      " -I/home/winter/Vc_head_install/include "
      "-I/home/winter/boost_1_63_0_install/include");

  //  #define L3_X 420 // max 2 L3 par set to 1024 (rest 512)
  // #define L3_Y 256
  // #define L3_K_STEP 256
  // #define L2_X 70 // max 2 L2 par set to 128 (rest 64)
  // #define L2_Y 64
  // #define L2_K_STEP 128
  // #define L1_X 35 // max all L1 par set to 32
  // #define L1_Y 16
  // #define L1_K_STEP 64
  // #define X_REG 5 // cannot be changed!
  // #define Y_REG 8 // cannot be changed!

  // autotune::combined_kernel.add_parameter("L3_X", {"210", "420"});
  // autotune::combined_kernel.add_parameter("L3_Y", {"128", "256"});
  // autotune::combined_kernel.add_parameter("L3_K_STEP", {"256"});
  autotune::combined_kernel.add_parameter("L2_X", {"70"});
  autotune::combined_kernel.add_parameter("L2_Y", {"64"});
  autotune::combined_kernel.add_parameter("L2_K_STEP", {"128"});
  autotune::combined_kernel.add_parameter("L1_X", {"10", "35", "70"});
  autotune::combined_kernel.add_parameter("L1_Y", {"8", "16", "32"});
  autotune::combined_kernel.add_parameter("L1_K_STEP", {"32", "64", "128"});

  // autotune::combined_kernel.add_parameter("L3_X", {"420"});
  // autotune::combined_kernel.add_parameter("L3_Y", {"256"});
  // autotune::combined_kernel.add_parameter("L3_K_STEP", {"256"});
  // autotune::combined_kernel.add_parameter("L2_X", {"70"});
  // autotune::combined_kernel.add_parameter("L2_Y", {"64"});
  // autotune::combined_kernel.add_parameter("L2_K_STEP", {"128"});
  // autotune::combined_kernel.add_parameter("L1_X", {"35"});
  // autotune::combined_kernel.add_parameter("L1_Y", {"16"});
  // autotune::combined_kernel.add_parameter("L1_K_STEP", {"64"});

  autotune::combined_kernel.set_source_dir("src/variants/combined_kernel");

  // std::vector<double> C_return(N * N, 0.0);
  // std::fill(C_return.begin(), C_return.end(), 0.0);

  std::vector<size_t> optimal_parameter_indices(
      autotune::combined_kernel.get_parameters().size(), 0.0);

  // autotune::combined_kernel.create_parameter_file(optimal_parameter_indices);
  // autotune::combined_kernel.compile();

  // if (autotune::combined_kernel.is_valid_parameter_combination()) {
  //   std::cout << "parameter combination is valid" << std::endl;
  // } else {
  //   std::cout << "parameter combination is NOT valid" << std::endl;
  // }

  // std::vector<size_t> optimal_parameter_indices;
  double tune_kernel_duration_temp;

  auto test_result = [](const std::vector<double> &C) -> bool {
    // std::cout << "C_reference:" << std::endl;
    // print_matrix_host(N, C_reference);
    // std::cout << "C mine:" << std::endl;
    // print_matrix_host(N, C);
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
  optimal_parameter_indices = autotune::combined_kernel.tune(
      autotune::tuner::bruteforce, test_result, m.N_org, m.X_size, m.Y_size,
      m.K_size, m.A, m.B, m.repetitions, tune_kernel_duration_temp);

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
