#!/bin/bash -e
set -x

mkdir build
cd build
cmake -DBOOST_ROOT="$Boost_ROOT" -DCMAKE_PREFIX_PATH="$HPX_ROOT" ../
make -j2 VERBOSE=1
cd ..
