# MatrixMultiplicationHPX

## What is this about?

MatrixMultiplicationHPX is a small project that demonstrates the usability of [HPX](https://github.com/STEllAR-GROUP/hpx) for fast square matrix multiplication. The repository implements algorithms that reach up to about 70% peak performance on a modern Intel processor. There are also distributed algorithms and a large selection of different approaches.

This is a demonstrator project to show that modern C++ together with the library HPX and the vectorization library [Vc](https://github.com/VcDevel/Vc) enable a relatively easy to write, easy to maintain, but still very fast implementation of a important linear algebra algorithm.

We additionally provide algorithm variants with different optimizations, so that the optimization project can be tracked throughout the optimizations and the benefit of individual optimizations becomes clearer. For more details on this see the subproject found in the folder otherExperiments, which includes a matrix multiplication that starts with no optimizations and is then optimized up to ~50% peak performance.

## Compilation

Needs a installation of HPX with datapar enabled, requires the build tool Scons to be installed. To configure the project make Scons find the HPX installation by specifying two paths in custom.py (see custom.py_example). Then type "scons" in the root directory to build the project.

## Usage

```
./release/matrix_multiply --help
Usage: matrix_multiply [options]:
  --n-value arg (=4)                    n value for the square matrices, should
                                        be a power of 2, arbitrary sized square
                                        matrices work with some implementations
  --transposed arg (=1)                 use a transposed matrix for B
  --repetitions arg (=1)                how often should the operation be
                                        repeated (for averaging timings)
  --verbose arg (=0)                    set to 1 for some status information,
                                        set to 2 more output
  --block-result arg (=128)             block size in the result matrix (width
                                        of the band of the band matrix
                                        multiplication), set to 0 to disable
  --block-input arg (=128)              chunks the band of the band matrix
                                        multiplication
  --check arg (=0)                      check result against a naive and slow
                                        matrix-multiplication implementation
  --algorithm arg (=single)             select algorithm: single,
                                        pseudodynamic, algorithms, looped,
                                        semi, combined, kernel_test,
                                        kernel_tiled
  --min-work-size arg (=256)            pseudodynamic algorithm: minimum work
                                        package size per node
  --max-work-difference arg (=10000)    pseudodynamic algorithm: maximum
                                        tolerated load inbalance in matrix
                                        components assigned
  --max-relative-work-difference arg (=0.050000000000000003)
                                        pseudodynamic algorithm: maximum
                                        relative tolerated load inbalance in
                                        matrix components assigned, in percent
  --help                                display help
```

## Some performance results

All results obtained on a single i7 6700k

# Node-level low-level HPX component-based variants:

```
./release/matrix_multiply --n-value=8192 --check=False --algorithm=single --block-result=128 --block-input=128 --hpx:threads=4
using parallel single node algorithm
[N = 8192] total time: 12.7411s
[N = 8192] average time per run: 12.7411s (repetitions = 1)
[N = 8192] performance: 86.2967Gflops (average across repetitions)
```

# Parallel algorithm based variants:

```
./release/matrix_multiply --n-value=8192 --check=False --algorithm=combined --block-result=128 --block-input=128 --hpx:threads=4
duration inner: 6.76058s
[X_size = 8400, Y_size = 8192, K_size = 8192] inner performance: 166.765Gflops (average across repetitions)
[N = 8192] total time: 8.5711s
[N = 8192] average time per run: 8.5711s (repetitions = 1)
[N = 8192] performance: 128.281Gflops (average across repetitions)
```

Ther inner performance is relevant, which excludes the matrix creation overhead. This is required, because of the fast matrix processing.

```
./release/matrix_multiply --n-value=8192 --check=False --algorithm=kernel_tiled --block-result=128 --block-input=128 --hpx:threads=4
duration inner: 6.45054s
[X_size = 8400, Y_size = 8192, K_size = 8192] inner performance: 174.781Gflops (average across repetitions)
non-HPX [N = 8192] total time: 6.45054s
non-HPX [N = 8192] average time per run: 6.45054s (repetitions = 1)
[N = 8192] performance: 170.453Gflops (average across repetitions)
```

The inner performance is relevant for this algorithm as well. Fastest variant.

# Distributed HPX component-based variant:

```
(example run with only a single node)
./release/matrix_multiply --n-value=8192 --check=False --algorithm=pseudodynamic --block-result=128 --block-input=128 --hpx:threads=4
using pseudodynamic distributed algorithm
[N = 8192] total time: 13.0368s
[N = 8192] average time per run: 13.0368s (repetitions = 1)
[N = 8192] performance: 84.3391Gflops (average across repetitions)
```

# Somewhat optimized OpenMP-based reference implementation:

```
./release/matrix_multiply --n-value=8192 --check=False --algorithm=kernel_test --block-result=128 --block-input=128 --hpx:threads=4
non-HPX [N = 8192] total time: 11.5435s
non-HPX [N = 8192] average time per run: 11.5435s (repetitions = 1)
[N = 8192] performance: 95.2494Gflops (average across repetitions)
```

## Testing

Unit tests are built alongside the application.
