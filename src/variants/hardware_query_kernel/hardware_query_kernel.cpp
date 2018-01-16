#include <Vc/Vc>
using Vc::double_v;

extern "C" uint64_t hardware_query_kernel() { return double_v::size(); }
