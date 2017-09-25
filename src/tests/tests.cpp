#include <sstream>
#include <hpx/hpx_start.hpp>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE cpp_hpx_vc_matrix_multiplication
#include <boost/test/unit_test.hpp>

#include "tests.hpp"

void start_hpx_with_threads(size_t threads) {
  std::stringstream s_threads;
  s_threads << "--hpx:threads=" << threads;
#ifdef DISABLE_BIND_FOR_CIRCLE_CI
  std::stringstream s_bind;
  s_bind << " --hpx:bind=none";
  std::cout << "info: disabling bind for Circle CI" << std::endl;
#endif
  std::string hpx_threads(s_threads.str());
  // std::string hpx_threads = "--hpx:threads=4";
  std::vector<char *> argv_hpx;
  argv_hpx.push_back(boost::unit_test::framework::master_test_suite().argv[0]);
  argv_hpx.push_back(const_cast<char *>(hpx_threads.c_str()));
#ifdef DISABLE_BIND_FOR_CIRCLE_CI
  std::string hpx_bind(s_bind.str());
  argv_hpx.push_back(const_cast<char *>(hpx_bind.c_str()));
#endif
  char **argv_hpx_ptr = argv_hpx.data();
#ifdef DISABLE_BIND_FOR_CIRCLE_CI
  hpx::start(3, argv_hpx_ptr);
#else
  hpx::start(2, argv_hpx_ptr);
#endif
}
