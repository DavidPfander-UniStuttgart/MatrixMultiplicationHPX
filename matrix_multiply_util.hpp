#pragma once

#include <iostream>
#include <hpx/include/iostreams.hpp>

template<typename T>
void print_matrix(size_t N, std::vector<T> m) {
  for (std::uint64_t i = 0; i < N; i++) {
    for (std::uint64_t j = 0; j < N; j++) {
      if (j > 0) {
	hpx::cout << ", ";
      }
	  hpx::cout << m.at(i * N + j);
    }
      hpx::cout << std::endl << hpx::flush;
  }
}
