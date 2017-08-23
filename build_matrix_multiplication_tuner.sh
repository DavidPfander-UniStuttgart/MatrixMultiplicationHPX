#!/bin/bash

if [ -z "$1" ]; then
    echo "info: no argument for build parallelism supplied, setting to 1"
    export PARALLEL_BUILD=1
else
    export PARALLEL_BUILD=$1
fi

echo "parallel build (-j for make): $PARALLEL_BUILD"

source source-me.sh

./scripts/build-jemalloc.sh
./scripts/build-boost.sh
./scripts/build-Vc.sh
./scripts/build-hpx.sh
./scripts/build-AutoTuneTMP.sh
./build-MatrixMultiplicationHPX.sh
