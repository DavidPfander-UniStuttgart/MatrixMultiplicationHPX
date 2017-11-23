#!/bin/bash

if [[ -z "$1" || "$1" != "circle" ]]; then
    #use all available CPUs
    export PARALLEL_BUILD=$((`lscpu -p=cpu | wc -l`-4))
else
    # circle uses special parallelism settings in the build scripts to maximize performance per scripts
    # all 4 threads ins't always possible, because of memory limititations (4G)
    export PARALLEL_BUILD=1
fi

echo "parallel build (-j for make): $PARALLEL_BUILD"


if [[ "$MATRIX_MULTIPLICATION_TARGET" = "knl" ]]; then
    module load craype-mic-knl
    module switch PrgEnv-cray/6.0.3 PrgEnv-gnu
    module load CMake/3.8.1
    export CC=gcc
    export CXX=g++
    export CXX_FLAGS="-fPIC -march=knl -mtune=knl -ffast-math"
else
    export CC=gcc
    export CXX=g++
    export CXX_FLAGS="-march=native -mtune=native"
fi
export Vc_ROOT=$PWD/Vc_install
# export Boost_ROOT=$PWD/boost_install
# export BOOST_ROOT=$Boost_ROOT
export BOOST_ROOT=$PWD/boost_install
# export HPX_ROOT=$PWD/hpx_install
export VC_ROOT=$PWD/Vc_install
export CPPJIT_ROOT=$PWD/cppjit_install
export AUTOTUNETMP_ROOT=$PWD/AutoTuneTMP_install
export MatrixMultiplicationHPX_ROOT=$PWD/MatrixMultiplicationHPX_install
export JEMALLOC_ROOT=$PWD/jemalloc_install

export LD_LIBRARY_PATH=$PWD/boost_install/lib:$PWD/hpx_install/lib:$PWD/Vc_install/lib

export matrix_multiplication_source_me_sourced=1

if [[ ! -z "$1" && "$1" = "circle" ]]; then
    export PATH=$PWD/cmake/bin:$PATH
fi
