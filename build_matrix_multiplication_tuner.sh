#!/bin/bash
set -x
set -e

if [[ -z "$1" || $1 != "circle" ]]; then
    if [ -z "$2" ]; then
        # see special case for Circle CI below
        echo "info: no argument for build parallelism supplied, setting to 1"
        export PARALLEL_BUILD=1
    else
        export PARALLEL_BUILD=$2
    fi
fi
# circle uses special parallelism settings in the build scripts to maximize performance per scripts
# all 4 threads ins't always possible, because of memory limititations (4G)

if [ -z "$1" ]; then
    echo "no target machine specified, building for native"
    export MATRIX_MULTIPLICATION_TARGET="native"    
elif [ "$1" = "knl" ]; then
    export MATRIX_MULTIPLICATION_TARGET="knl"
elif [ "$1" = "circle" ]; then
    export MATRIX_MULTIPLICATION_TARGET="circle"
fi


echo "parallel build (-j for make): $PARALLEL_BUILD"

source source-me.sh

./scripts/build-jemalloc.sh
./scripts/build-boost.sh
./scripts/build-Vc.sh
./scripts/build-hpx.sh
./scripts/build-AutoTuneTMP.sh
./build-MatrixMultiplicationHPX.sh
