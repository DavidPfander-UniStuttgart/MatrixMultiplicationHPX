#define BOOST_TEST_DYN_LINK

// global test variables are defined here
#include "util/create_identity_matrix.hpp"
#include "util/create_random_matrix.hpp"

#include "reference_kernels/naive.hpp"
#include "util/util.hpp"

#include <vector>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(test_naive)

BOOST_AUTO_TEST_CASE(random_matrices) {

  size_t N = 2;

  std::vector<double> A = {2., 5., 1., 3.};
  std::vector<double> B = {3., -5, -1, 2.};

  std::vector<double> C_reference = util::create_identity_matrix<double>(2);

  std::vector<double> C = naive_matrix_multiply(N, A, B);

  std::cout << "N: " << N << std::endl;
  std::cout << "C:" << std::endl;
  print_matrix_host(N, C);

  std::cout << "C_reference:" << std::endl;
  print_matrix_host(N, C_reference);

  for (size_t i = 0; i < N * N; i++) {
    BOOST_CHECK_CLOSE(C[i], C_reference[i], 1E-10);
  }
}

BOOST_AUTO_TEST_SUITE_END()
