#pragma once

#include <cstddef>

struct matrix_multiply_work {
  size_t x;
  size_t y;
  size_t N;
  matrix_multiply_work(size_t x, size_t y, size_t N) : x(x), y(y), N(N) {}
};
