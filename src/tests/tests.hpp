#ifdef DEBUG
#error "Vc in debug mode will generate instructions of the wrong length, aborting"
#endif

#ifndef __clang__
#include <omp.h>
#else
// dummy if OpenMP is not enabled
int omp_get_max_threads(void) {
	return 1;
}
#endif

void start_hpx_with_threads(size_t threads);
