#include <sstream>
#include <hpx/hpx_start.hpp>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE cpp_hpx_vc_matrix_multiplication
#include <boost/test/unit_test.hpp>

#include "tests.hpp"


void start_hpx_with_threads(size_t threads) {
  std::stringstream s;
  s << "--hpx:threads=" << threads;
#ifdef DISABLE_BIND_FOR_CIRCLE_CI
  s << " --hpx:bind=none";
#endif
  std::string hpx_threads(s.str());
  // std::string hpx_threads = "--hpx:threads=4";
  std::vector<char *> argv_hpx;
  argv_hpx.push_back(boost::unit_test::framework::master_test_suite().argv[0]);
  argv_hpx.push_back(const_cast<char *>(hpx_threads.c_str()));
  char **argv_hpx_ptr = argv_hpx.data();
  hpx::start(2, argv_hpx_ptr);
}
