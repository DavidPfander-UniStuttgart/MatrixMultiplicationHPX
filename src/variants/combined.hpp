#pragma once

#include "autotune/autotune.hpp"
#include "autotune/parameter.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

AUTOTUNE_DECLARE_KERNEL(void(std::size_t, std::size_t, std::size_t, std::size_t,
                             std::vector<double> &, std::vector<double> &,
                             std::vector<double> &, size_t, double &),
                        combined_kernel)

namespace combined {

class combined {

public:
  std::size_t N_org;
  std::size_t X_size;
  std::size_t Y_size;
  std::size_t K_size;

  std::vector<double> A;
  std::vector<double> B;

  uint64_t repetitions;
  uint64_t verbose;

  void verify_blocking_setup();

public:
  combined(size_t N, std::vector<double> &A, std::vector<double> &B,
           uint64_t repetitions, uint64_t verbose);

  std::vector<double> matrix_multiply(double &duration);
};
}
