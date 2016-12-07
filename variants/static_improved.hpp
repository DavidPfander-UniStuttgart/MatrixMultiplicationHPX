#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

#include "matrix_multiply_work.hpp"

namespace multiply_components {

// uses round-robin distribution scheme, granularity of distribution is determined by the number of nodes
class static_improved {
private:
    size_t N;
    std::vector<double> &A;
    std::vector<double> &B;
    bool transposed;
    uint64_t block_input;
    std::vector<double> C;
    size_t block_result;

    uint64_t min_work_size;
    uint64_t max_work_difference;
    double max_relative_work_difference;

    uint64_t repetitions;
    uint64_t verbose;
public:
    static_improved(size_t N, std::vector<double> &A,
            std::vector<double> &B, bool transposed, uint64_t block_input,
            size_t block_result, uint64_t min_work_size,
            uint64_t max_work_difference, double max_relative_work_difference,
            uint64_t repetitions, uint64_t verbose) :
            N(N), A(A), B(B), transposed(transposed), block_input(block_input), C(
                    N * N), block_result(block_result), min_work_size(
                    min_work_size), max_work_difference(max_work_difference), max_relative_work_difference(
                    max_relative_work_difference), repetitions(repetitions), verbose(
                    verbose) {
    }

    void print_schedule(
            std::vector<std::vector<matrix_multiply_work>> &all_work);

    void insert_submatrix(const std::vector<double> &submatrix,
            const matrix_multiply_work &w);

    bool fulfills_constraints(std::vector<uint64_t> &total_work);

    std::vector<std::vector<matrix_multiply_work>> create_schedule(
            size_t num_localities);

    std::vector<double> matrix_multiply();
};

}
