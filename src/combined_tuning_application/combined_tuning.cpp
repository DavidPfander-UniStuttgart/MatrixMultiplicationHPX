#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/util.hpp>

#include "autotune/autotune.hpp"
#include "autotune/parameter.hpp"

#include "reference_kernels/naive.hpp"
#include "util/create_random_matrix.hpp"
#include "util/matrix_multiplication_exception.hpp"
#include "variants/combined.hpp"

#include <random>

int hpx_main() {

  std::vector<double> C;
  std::uint64_t N = 4096;
  bool transposed = false;
  size_t repetitions = 3;
  bool verbose = false;

  // create matrices A, B
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  auto myRand = std::bind(distribution, generator);

  std::vector<double> A = util::create_random_matrix<double>(N);
  std::vector<double> B = util::create_random_matrix<double>(N);

  std::cout << "calculating reference solution..." << std::flush;
  std::vector<double> C_reference = naive_matrix_multiply(N, A, B);
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
      "-I /home/winter/git/AutoTuneTMP/src -I src/variants/ -I "
      "/home/winter/hpx_install_with_symbols/include -I "
      "/home/winter/hpx_install_with_symbols/include/hpx/external -DNDEBUG "
      "-std=c++14 -march=native -mtune=native -O3 -ffast-math "
      "-DHPX_APPLICATION_EXPORTS "
      "-DHPX_ENABLE_ASSERT_HANDLER -I/home/winter/Vc_head_install/include "
      "-I/home/winter/boost_1_63_0_install/include");

  //  #define L3_X 420 // max 2 L3 par set to 1024 (rest 512)
  autotune::combined_kernel.add_parameter("L3_X", {"210", "420"});
  // #define L3_Y 256
  autotune::combined_kernel.add_parameter("L3_Y", {"128", "256"});
  //#define L3_K_STEP 256
  autotune::combined_kernel.add_parameter("L3_K_STEP", {"256"});
  //#define L2_X 70 // max 2 L2 par set to 128 (rest 64)
  autotune::combined_kernel.add_parameter("L2_X", {"70"});
  // #define L2_Y 64
  autotune::combined_kernel.add_parameter("L2_Y", {"64"});
  // #define L2_K_STEP 128
  autotune::combined_kernel.add_parameter("L2_K_STEP", {"128"});
  // #define L1_X 35 // max all L1 par set to 32
  autotune::combined_kernel.add_parameter("L1_X", {"35"});
  // #define L1_Y 16
  autotune::combined_kernel.add_parameter("L1_Y", {"16"});
  // #define L1_K_STEP 64
  autotune::combined_kernel.add_parameter("L1_K_STEP", {"64"});
  // #define X_REG 5 // cannot be changed!
  // #define Y_REG 8 // cannot be changed!

  autotune::combined_kernel.set_source_dir("src/variants/combined_kernel");

  std::vector<double> C_return(N * N, 0.0);
  std::fill(C_return.begin(), C_return.end(), 0.0);

  // std::vector<size_t> optimal_parameter_indices(
  //     autotune::combined_kernel.get_parameters().size(), 0.0);

  // autotune::combined_kernel.create_parameter_file(optimal_parameter_indices);
  // autotune::combined_kernel.compile();

  // if (autotune::combined_kernel.is_valid_parameter_combination()) {
  //   std::cout << "parameter combination is valid" << std::endl;
  // } else {
  //   std::cout << "parameter combination is NOT valid" << std::endl;
  // }

  std::vector<size_t> optimal_parameter_indices;
  double tune_kernel_duration_temp;

  auto test_result = [&C_reference, N](const std::vector<double> &C) -> bool {
    for (size_t i = 0; i < N * N; i++) {
      if (fabs(C[i] - C_reference[i]) >= 1E-8) {
        return false;
      }
    }
    return true;
  };

  std::cout << "----------------------- starting tuning -----------------------"
            << std::endl;
  optimal_parameter_indices = autotune::combined_kernel.tune(
      test_result, m.N_org, m.X_size, m.Y_size, m.K_size, m.A, m.B,
      m.repetitions, tune_kernel_duration_temp);

  std::cout << "----------------------- end tuning -----------------------"
            << std::endl;
  std::cout << "optimal parameter values:" << std::endl;
  autotune::combined_kernel.print_values(optimal_parameter_indices);

  autotune::combined_kernel.create_parameter_file(optimal_parameter_indices);

  autotune::combined_kernel.compile();

  double inner_duration;
  C = m.matrix_multiply(inner_duration);
  return hpx::finalize();
}

int main(int argc, char **argv) { int return_value = hpx::init(argc, argv); }
