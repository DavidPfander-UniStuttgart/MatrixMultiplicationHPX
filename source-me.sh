#!/bin/bash

if [[ -z "$CONFIGURATION_DONE" ]]; then
    if [[ -z "$1" || "$1" != "circle" ]]; then
        #use all available CPUs
        export PARALLEL_BUILD=$((`lscpu -p=cpu | wc -l`-4))
    else
        # circle uses special parallelism settings in the build scripts to maximize performance per scripts
        # all threads ins't always possible, because of memory limititations (4G)
        export PARALLEL_BUILD=4
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
    export CONFIGURATION_DONE=true
else
    echo "MatrixMultiplicationHPX: configuration already done, skipping..."
fi

# source the remaining dependencies recursively
source $(readlink -f $(dirname "$BASH_SOURCE"))/AutoTuneTMP/source-me.sh

export MatrixMultiplicationHPX_ROOT=$PWD/MatrixMultiplicationHPX_install
export LD_LIBRARY_PATH=$PWD/AutoTuneTMP/boost_install/lib:$PWD/AutoTuneTMP/Vc_install/lib
export matrix_multiplication_source_me_sourced=1
