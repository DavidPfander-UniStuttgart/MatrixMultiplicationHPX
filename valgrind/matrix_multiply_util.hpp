#pragma once

#include <iostream>

template<typename T>
void print_matrix_host(size_t N, std::vector<T> m) {
	for (std::uint64_t i = 0; i < N; i++) {
		for (std::uint64_t j = 0; j < N; j++) {
			if (j > 0) {
				std::cout << ", ";
			}
			std::cout << m.at(i * N + j);
		}
		std::cout << std::endl;
	}
}

template<typename T>
void print_matrix_transposed_host(size_t N, std::vector<T> m) {
	for (std::uint64_t i = 0; i < N; i++) {
		for (std::uint64_t j = 0; j < N; j++) {
			if (j > 0) {
				std::cout << ", ";
			}
			std::cout << m.at(j * N + i);
		}
		std::cout << std::endl;
	}
}
