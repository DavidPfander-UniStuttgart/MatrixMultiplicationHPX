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

./scripts/build-AutoTuneTMP.sh

# need this, otherwise have to push arguments to called scripts
source source-me.sh

./build-MatrixMultiplicationHPX.sh
