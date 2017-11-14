#!/bin/bash
set -x
set -e

if [ -z "$1" ]; then
    echo "no target machine specified, building for native"
    export MATRIX_MULTIPLICATION_TARGET="native"    
elif [ "$1" = "knl" ]; then
    export MATRIX_MULTIPLICATION_TARGET="knl"
elif [ "$1" = "circle" ]; then
    export MATRIX_MULTIPLICATION_TARGET="circle"
fi

# need this, otherwise have to push arguments to called scripts
source source-me.sh

if [[ ! -z "$1" && "$1" = "circle" ]]; then
    ./scripts/build-cmake.sh
fi

./scripts/build-jemalloc.sh
./scripts/build-boost.sh
./scripts/build-Vc.sh
# ./scripts/build-hpx.sh
./scripts/build-AutoTuneTMP.sh
./build-MatrixMultiplicationHPX.sh
