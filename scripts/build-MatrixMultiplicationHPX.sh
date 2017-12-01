#!/bin/bash -e
set -x
set -e

if [[ -z ${matrix_multiplication_source_me_sourced} ]]; then
    source source-me.sh
fi

git pull

mkdir -p build_RelWithDebInfo
cd build_RelWithDebInfo

# detection of Vc doesn't work with a relative path
cmake -DAUTOTUNETMP_ROOT=${AUTOTUNETMP_ROOT} -DVc_ROOT=${Vc_ROOT} -DCPPJIT_ROOT=${CPPJIT_ROOT} -DBOOST_ROOT=${BOOST_ROOT} -DCMAKE_BUILD_TYPE=RelWithDebInfo ../

make -j${PARALLEL_BUILD} VERBOSE=1
cd ../
