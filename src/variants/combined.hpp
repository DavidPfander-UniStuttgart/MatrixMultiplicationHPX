#pragma once

#include "autotune/autotune.hpp"
#include "autotune/parameter.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

AUTOTUNE_DECLARE_KERNEL(std::vector<double>(std::size_t, std::vector<double> &,
                                            std::vector<double> &, size_t,
                                            double &, double &),
                        combined_kernel)

namespace combined {

class combined {

public:
  std::size_t N_org;

  std::vector<double> A_org;
  std::vector<double> B_org;

  uint64_t repetitions;
  uint64_t verbose;

public:
  combined(size_t N, std::vector<double> &A_org, std::vector<double> &B_org,
           uint64_t repetitions, uint64_t verbose);

  ~combined();

  std::vector<double> matrix_multiply(double &duration, double &gflops_kernel,
                                      bool set_default_parameters = true);
};
} // namespace combined
