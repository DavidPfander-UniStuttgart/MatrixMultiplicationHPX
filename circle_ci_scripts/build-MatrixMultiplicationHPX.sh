#!/bin/bash -e
set -x

mkdir build
cd build
cmake -DDISABLE_BIND_FOR_CIRCLE_CI=ON -DBOOST_ROOT="$Boost_ROOT" -DCMAKE_PREFIX_PATH="$HPX_ROOT" -DCMAKE_BUILD_TYPE=release ../
make -j2 VERBOSE=1
cd ..
