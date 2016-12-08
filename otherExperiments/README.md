All results obtained on Intel i7 6700K
The results were not checked carefully, some implementation variants seem to be buggy (blocked4, gpuasm)
n = 4096 (set in benchmark.cpp, should be increased on modern hardware for reliable results)

The fastest (propably) correct implementation reaches up to ~50% peak performance.

```
./benchmark naive
total flops 2n^3 -> 137438953472
gflop: 137.438953472
seconds: 200.649
0.684972033113 Gflops
check: 8192
```

```
./benchmark transposed
total flops 2n^3 -> 137438953472
gflop: 137.438953472
seconds: 73.292
1.87522449206 Gflops
check: 8192
```

```
./benchmark counter
total flops 2n^3 -> 137438953472
gflop: 137.438953472
seconds: 5.665
24.2610685741 Gflops
check: 8192
```

```
./benchmark asm
total flops 2n^3 -> 137438953472
gflop: 137.438953472
seconds: 5.665
24.2610685741 Gflops
check: 8192
```

```
./benchmark blocked2
total flops 2n^3 -> 137438953472
gflop: 137.438953472
seconds: 2.577
53.3329272301 Gflops
check: 8192
```

```
./benchmark gpu
total flops 2n^3 -> 137438953472
gflop: 137.438953472
seconds: 2.571
53.457391471 Gflops
check: 8192
```

```
./benchmark intrinLoop
total flops 2n^3 -> 137438953472
gflop: 137.438953472
seconds: 1.212
113.398476462 Gflops
check: 8192
```
