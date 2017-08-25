#define BOOST_TEST_DYN_LINK

#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>

#include "tests.hpp"
#include <boost/test/unit_test.hpp>

#include "util/create_identity_matrix.hpp"
#include "util/create_random_matrix.hpp"
#include "util/pattern_matrices.hpp"
#include "util/transpose_matrix.hpp"
// global test variables are defined here
#include "reference_kernels/kernel_test.hpp"
#include "reference_kernels/kernel_tiled.hpp"
#include "reference_kernels/naive.hpp"
#include "test_hpx_main.hpp"
#include "util/util.hpp"


BOOST_AUTO_TEST_SUITE(test_semi)

BOOST_AUTO_TEST_CASE(apply_inverse_4) {

  using namespace hpx_parameters;

  N = 4;
  A = {-4., 5.,  0.,  -3., //
       -1., 2.,  0.,  2.,  //
       -8., -3., 8.,  -8., //
       -4., 3.,  -4., -1};

  B = {0.1347150259,  -0.2124352332, -0.0829015544, -0.1658031088, //
       0.2176165803,  0.0414507772,  -0.0569948187, -0.1139896373, //
       0.0660621762,  0.1554404145,  0.0362694301,  -0.1774611399, //
       -0.1502590674, 0.3523316062,  0.0155440415,  0.0310880829};
	B = util::transpose_matrix(N, B);

  C = std::vector<double>();
  C_reference = util::create_identity_matrix<double>(N);

  algorithm = "semi";
  verbose = false;
  check = true;
  // is B transposed, relevant for some (reference) algorithm
  transposed = true;

  block_input = 1;
  block_result = 1;

  duration = 0.0; // write variable
  repetitions = 1;
  is_root_node = false;

  min_work_size = 0;                  // unused
  max_work_difference = 0;            // unused
  max_relative_work_difference = 0.0; // unused

  // Initialize HPX, run hpx_main.
  start_hpx_with_threads(omp_get_max_threads());

  // Wait for hpx::finalize being called.
  hpx::stop();

  // std::cout << "N: " << N << std::endl;
  // std::cout << "C:" << std::endl;
  // print_matrix_host(N, C);

  // std::cout << "C_reference:" << std::endl;
  // print_matrix_host(N, C_reference);

  for (size_t i = 0; i < N * N; i++) {
    BOOST_CHECK_SMALL(fabs(C[i] - C_reference[i]), 1E-8);
  }
}

BOOST_AUTO_TEST_CASE(apply_inverse_8) {

  using namespace hpx_parameters;

  N = 8;

  A = {0.,  0.,  0.,  -1., -1., 1.,  1.,  0.,  //
       -1., -2., 0.,  0.,  0.,  -4., -1., 0.,  //
       0.,  0.,  0.,  -2., 0.,  0.,  -4., 0.,  //
       -3., 0.,  4.,  -2., 0.,  -7., -4., -3., //
       0.,  2.,  0.,  1.,  -3., -2., -1., -1., //
       0.,  0.,  0.,  4.,  0.,  3.,  0.,  2.,  //
       0.,  -1., -4., -2., 0.,  3.,  -5., 0.,  //
       1.,  0.,  1.,  0.,  -6., 1.,  -5., 0};

  B = {-0.945977686435702, -0.125660598943042, -0.092190252495597,
       -0.206106870229008, -0.196711685261304, -0.407516147974163,
       -0.142102172636524, 0.256018790369935, //
       0.237815619495009,  -0.087962419260129, 0.435466823253083,
       -0.044274809160305, 0.362301820317088,  0.114738696418086,
       -0.099471520845567,
       -0.220786846741045, //
       -0.059894304169114, -0.121550205519671, 0.036993540810335,
       0.076335877862595,  -0.194950088079859, 0.017028772753964,
       -0.146799765120376,
       0.107457428068115, //
       -0.405167351732237, -0.057545507927187, -0.30857310628303,
       0.045801526717557,  -0.024662360540223, 0.056371109806224,
       0.065766294773928,
       0.079859072225484, //
       -0.325308279506753, -0.095478567234292, 0.042102172636524,
       0.010687022900763,  -0.098062243100411, -0.03300058719906,
       -0.005167351732237,
       -0.063417498532002, //
       0.066940692894891,  -0.181796829125073, -0.170757486788021,
       0.079389312977099,  -0.135055783910746, 0.051556077510276,
       0.093482090428655,
       0.056371109806224, //
       0.202583675866119,  0.028772753963594,  -0.095713446858485,
       -0.022900763358779, 0.012331180270112,  -0.028185554903112,
       -0.032883147386964,
       -0.039929536112742, //
       0.709923664122137,  0.387786259541985,  0.873282442748091,
       -0.210687022900763, 0.251908396946565,  0.309923664122138,
       -0.27175572519084,  -0.244274809160305};
	B = util::transpose_matrix(N, B);

  C = std::vector<double>();
  C_reference = util::create_identity_matrix<double>(N);

  algorithm = "semi";
  verbose = false;
  check = true;
  // is B transposed, relevant for some (reference) algorithm
  transposed = true;

  block_input = 1;
  block_result = 1;

  duration = 0.0; // write variable
  repetitions = 1;
  is_root_node = false;

  min_work_size = 0;                  // unused
  max_work_difference = 0;            // unused
  max_relative_work_difference = 0.0; // unused

  // Initialize HPX, run hpx_main
  start_hpx_with_threads(omp_get_max_threads());

  // Wait for hpx::finalize being called.
  hpx::stop();

  // std::cout << "N: " << N << std::endl;
  // std::cout << "C:" << std::endl;
  // print_matrix_host(N, C);

  // std::cout << "C_reference:" << std::endl;
  // print_matrix_host(N, C_reference);

  for (size_t i = 0; i < N * N; i++) {
    BOOST_CHECK_SMALL(fabs(C[i] - C_reference[i]), 1E-8);
  }
}

BOOST_AUTO_TEST_CASE(random_matrices_4) {

  using namespace hpx_parameters;

  N = 4;

  A = util::create_random_matrix<double>(N);
  B = util::create_random_matrix<double>(N);

  C = std::vector<double>();
  C_reference = std::vector<double>();

  algorithm = "semi";
  verbose = false;
  check = true;
  // is B transposed, relevant for some (reference) algorithm
  transposed = true;

  block_input = 1;
  block_result = 1;

  duration = 0.0; // write variable
  repetitions = 1;
  is_root_node = false;

  min_work_size = 0;                  // unused
  max_work_difference = 0;            // unused
  max_relative_work_difference = 0.0; // unused

  // Initialize HPX, run hpx_main.
  start_hpx_with_threads(omp_get_max_threads());

  // Wait for hpx::finalize being called.
  hpx::stop();

  if (!transposed) {
    C_reference = naive_matrix_multiply(N, A, B);
  } else {
    C_reference = naive_matrix_multiply_transposed(N, A, B);
  }

  for (size_t i = 0; i < N * N; i++) {
    BOOST_CHECK_SMALL(fabs(C[i] - C_reference[i]), 1E-8);
  }
}

BOOST_AUTO_TEST_CASE(random_matrices_8) {

  using namespace hpx_parameters;

  N = 8;

  A = util::create_random_matrix<double>(N);
  B = util::create_random_matrix<double>(N);

  C = std::vector<double>();
  C_reference = std::vector<double>();

  algorithm = "semi";
  verbose = false;
  check = true;
  // is B transposed, relevant for some (reference) algorithm
  transposed = true;

  block_input = 1;
  block_result = 1;

  duration = 0.0; // write variable
  repetitions = 1;
  is_root_node = false;

  min_work_size = 0;                  // unused
  max_work_difference = 0;            // unused
  max_relative_work_difference = 0.0; // unused

  // Initialize HPX, run hpx_main.
  start_hpx_with_threads(omp_get_max_threads());

  // Wait for hpx::finalize being called.
  hpx::stop();

  if (!transposed) {
    C_reference = naive_matrix_multiply(N, A, B);
  } else {
    C_reference = naive_matrix_multiply_transposed(N, A, B);
  }

  for (size_t i = 0; i < N * N; i++) {
    BOOST_CHECK_SMALL(fabs(C[i] - C_reference[i]), 1E-8);
  }
}

BOOST_AUTO_TEST_CASE(random_matrices_256) {

  using namespace hpx_parameters;

  N = 256;

  A = util::create_random_matrix<double>(N);
  B = util::create_random_matrix<double>(N);

  C = std::vector<double>();
  C_reference = std::vector<double>();

  algorithm = "semi";
  verbose = false;
  check = true;
  // is B transposed, relevant for some (reference) algorithm
  transposed = true;

  block_input = 128;
  block_result = 128;

  duration = 0.0; // write variable
  repetitions = 1;
  is_root_node = false;

  min_work_size = 0;                  // unused
  max_work_difference = 0;            // unused
  max_relative_work_difference = 0.0; // unused

  // Initialize HPX, run hpx_main.
  start_hpx_with_threads(omp_get_max_threads());

  // Wait for hpx::finalize being called.
  hpx::stop();

  if (!transposed) {
    C_reference = naive_matrix_multiply(N, A, B);
  } else {
    C_reference = naive_matrix_multiply_transposed(N, A, B);
  }

  for (size_t i = 0; i < N * N; i++) {
    BOOST_CHECK_SMALL(fabs(C[i] - C_reference[i]), 1E-8);
  }
}

BOOST_AUTO_TEST_CASE(random_matrices_512) {

  using namespace hpx_parameters;

  N = 512;

  A = util::create_random_matrix<double>(N);
  B = util::create_random_matrix<double>(N);

  C = std::vector<double>();
  C_reference = std::vector<double>();

  algorithm = "semi";
  verbose = false;
  check = true;
  // is B transposed, relevant for some (reference) algorithm
  transposed = true;

  block_input = 128;
  block_result = 128;

  duration = 0.0; // write variable
  repetitions = 1;
  is_root_node = false;

  min_work_size = 0;                  // unused
  max_work_difference = 0;            // unused
  max_relative_work_difference = 0.0; // unused

  // Initialize HPX, run hpx_main.
  start_hpx_with_threads(omp_get_max_threads());

  // Wait for hpx::finalize being called.
  hpx::stop();

  if (!transposed) {
    C_reference = naive_matrix_multiply(N, A, B);
  } else {
    C_reference = naive_matrix_multiply_transposed(N, A, B);
  }

  for (size_t i = 0; i < N * N; i++) {
    BOOST_CHECK_SMALL(fabs(C[i] - C_reference[i]), 1E-8);
  }
}

BOOST_AUTO_TEST_CASE(random_matrices_1024) {

  using namespace hpx_parameters;

  N = 1024;

  A = util::create_random_matrix<double>(N);
  B = util::create_random_matrix<double>(N);

  C = std::vector<double>();
  C_reference = std::vector<double>();

  algorithm = "semi";
  verbose = false;
  check = true;
  // is B transposed, relevant for some (reference) algorithm
  transposed = true;

  block_input = 128;
  block_result = 128;

  duration = 0.0; // write variable
  repetitions = 1;
  is_root_node = false;

  min_work_size = 0;                  // unused
  max_work_difference = 0;            // unused
  max_relative_work_difference = 0.0; // unused

  // Initialize HPX, run hpx_main.
  start_hpx_with_threads(omp_get_max_threads());

  // Wait for hpx::finalize being called.
  hpx::stop();

  if (!transposed) {
    C_reference = naive_matrix_multiply(N, A, B);
  } else {
    C_reference = naive_matrix_multiply_transposed(N, A, B);
  }

  for (size_t i = 0; i < N * N; i++) {
    BOOST_CHECK_SMALL(fabs(C[i] - C_reference[i]), 1E-8);
  }
}

BOOST_AUTO_TEST_SUITE_END()
