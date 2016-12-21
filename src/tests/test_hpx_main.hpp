#include <cstdint>
#include <string>
#include <vector>

namespace hpx_parameters {
extern std::vector<double> A;
extern std::vector<double> B;
extern std::vector<double> C;
extern std::vector<double> C_reference;
extern std::uint64_t N;
extern std::string algorithm;
extern std::uint64_t verbose;
extern bool check;
extern bool transposed;
extern uint64_t block_input;
extern size_t block_result;
// initialized via program_options defaults

extern double duration;
extern uint64_t repetitions;
// to skip printing and checking on all other nodes
extern bool is_root_node;

extern std::uint64_t min_work_size;
extern std::uint64_t max_work_difference;
extern double max_relative_work_difference;
}

int hpx_main(int argc, char *argv[]);
