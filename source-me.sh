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

export CC=gcc-7
export CXX=g++-7
if [[ "$MATRIX_MULTIPLICATION_TARGET" = "knl" ]]; then
    export CXX_FLAGS="-march=knl -mtune=knl"
else
    export CXX_FLAGS="-march=native -mtune=native"
fi
export Vc_ROOT=$PWD/Vc_install
# export Boost_ROOT=$PWD/boost_install
# export BOOST_ROOT=$Boost_ROOT
export BOOST_ROOT=$PWD/boost_install
export HPX_ROOT=$PWD/hpx_install
# not installed!
export AutoTuneTMP_ROOT=$PWD/AutoTuneTMP
export MatrixMultiplicationHPX_ROOT=$PWD/MatrixMultiplicationHPX_install
export JEMALLOC_ROOT=$PWD/jemalloc_install

export LD_LIBRARY_PATH=$PWD/boost_install/lib:$PWD/hpx_install/lib:$PWD/Vc_install/lib

export matrix_multiplication_source_me_sourced=1

if [[ ! -z "$1" && "$1" = "circle" ]]; then
    export PATH=/home/ubuntu/cmake/bin:$PATH
fi
