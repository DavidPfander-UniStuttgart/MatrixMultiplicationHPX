#!/bin/bash

source source-me.sh

./scripts/build-jemalloc.sh
./scripts/build-boost.sh
./scripts/build-Vc.sh
./scripts/build-hpx.sh
./scripts/build-AutoTuneTMP.sh
./scripts/build-MatrixMultiplicationHPX.sh
