#include "test_hpx_main.hpp"

#include <hpx/hpx_init.hpp>

#include "util/matrix_multiplication_exception.hpp"
#include "variants/algorithms.hpp"
#include "variants/combined.hpp"
#include "variants/looped.hpp"
#include "variants/proposal.hpp"
#include "variants/pseudodynamic.hpp"
#include "variants/semi.hpp"
#include "variants/single.hpp"

namespace hpx_parameters {
std::vector<double> A;
std::vector<double> B;
std::vector<double> C;
std::vector<double> C_reference;
std::uint64_t N;
std::string algorithm;
std::uint64_t verbose;
bool check;
bool transposed;
uint64_t block_input;
size_t block_result;
// initialized via program_options defaults

double duration;
uint64_t repetitions;
// to skip printing and checking on all other nodes
bool is_root_node;

std::uint64_t min_work_size;
std::uint64_t max_work_difference; // TODO: shouldn't this be a double?
double max_relative_work_difference;
}

int hpx_main(int argc, char *argv[]) {
  using namespace hpx_parameters;
  // Keep track of the time required to execute.
  hpx::util::high_resolution_timer t;

  if (hpx_parameters::algorithm.compare("single") == 0) {
    single::single m(N, A, B, transposed, block_input, block_result,
                     repetitions, verbose);
    C = m.matrix_multiply();
  } else if (hpx_parameters::algorithm.compare("pseudodynamic") == 0) {
    pseudodynamic::pseudodynamic m(
        N, A, B, transposed, block_input, block_result, min_work_size,
        max_work_difference, max_relative_work_difference, repetitions,
        verbose);
    C = m.matrix_multiply();
  } else if (hpx_parameters::algorithm.compare("algorithms") == 0) {
    if (!transposed) {
      throw util::matrix_multiplication_exception(
          "algorithm \"algorithms\" requires B to be transposed");
    }
    algorithms::algorithms m(N, A, B, block_input, block_result);
    C = m.matrix_multiply();
  } else if (hpx_parameters::algorithm.compare("looped") == 0) {
		if (!transposed) {
      throw util::matrix_multiplication_exception(
          "algorithm \"looped\" requires B to be transposed");
    }
    looped::looped m(N, A, B, block_result, block_input);
    C = m.matrix_multiply();
  } else if (hpx_parameters::algorithm.compare("semi") == 0) {
		if (!transposed) {
      throw util::matrix_multiplication_exception(
          "algorithm \"semi\" requires B to be transposed");
    }
    semi::semi m(N, A, B, block_result, block_input);
    C = m.matrix_multiply();
  } else if (hpx_parameters::algorithm.compare("combined") == 0) {
    if (transposed) {
      throw util::matrix_multiplication_exception(
          "algorithm \"combined\" doens't allow B to be transposed");
    }
    combined::combined m(N, A, B, repetitions, verbose);
    double inner_duration;
    C = m.matrix_multiply(inner_duration);
  } else if (hpx_parameters::algorithm.compare("proposal") == 0) {
    proposal::proposal m(N, A, B, transposed, block_result, block_input,
                         repetitions, verbose);
    double inner_duration;
    C = m.matrix_multiply(inner_duration);
  }

  duration = t.elapsed();
  // hpx::cout << "[N = " << N << "] total time: " << duration << "s" <<
  // std::endl
  //           << hpx::flush;
  // hpx::cout << "[N = " << N
  //           << "] average time per run: " << (duration / repetitions)
  //           << "s (repetitions = " << repetitions << ")" << std::endl
  //           << hpx::flush;

  // Any HPX application logic goes here...
  return hpx::finalize();
}
