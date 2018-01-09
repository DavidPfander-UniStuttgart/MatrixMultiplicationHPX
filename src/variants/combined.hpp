#pragma once

#include "autotune/autotune.hpp"
#include "autotune/parameter.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

AUTOTUNE_DECLARE_KERNEL(std::vector<double>(std::size_t, std::size_t,
                                            std::size_t, std::size_t,
                                            std::vector<double> &,
                                            std::vector<double> &, size_t,
                                            double &),
                        combined_kernel)

namespace combined {

// max 2 L3 par set to 1024 (rest 512)
constexpr uint64_t L3_X = 200;
constexpr uint64_t L3_Y = 256;
constexpr uint64_t L3_K_STEP = 256;

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

public:
  combined(size_t N, std::vector<double> &A, std::vector<double> &B,
           uint64_t repetitions, uint64_t verbose);

  ~combined();

  std::vector<double> matrix_multiply(double &duration);
};
}
