#!/bin/bash -e
set -x
set -e

if [ -z ${matrix_multiplication_source_me_sourced} ] ; then
    source source-me.sh
fi

# git pull

mkdir -p build_Release
cd build_Release

if [[ -z "$1" || "$1" != "circle" ]]; then
    CIRCLE_DISABLE_BIND=OFF
else
    CIRCLE_DISABLE_BIND=ON
fi

# detection of Vc doesn't work with a relative path
# cmake -DVc_ROOT="$MatrixMultiplicationHPX_ROOT" -DCMAKE_BUILD_TYPE=release ../
cmake -DDISABLE_BIND_FOR_CIRCLE_CI=$CIRCLE_DISABLE_BIND -DHPX_ROOT="$HPX_ROOT" -DAutoTuneTMP_ROOT=${AutoTuneTMP_ROOT} -DCMAKE_BUILD_TYPE=Release ../

# uses more than 4G with 4 threads (4G limit on Circle CI)
make -j${PARALLEL_BUILD} VERBOSE=1
cd ../

mkdir -p build_RelWithDebInfo
cd build_RelWithDebInfo

# detection of Vc doesn't work with a relative path
# cmake -DVc_ROOT="$MatrixMultiplicationHPX_ROOT" -DCMAKE_BUILD_TYPE=release ../
cmake -DHPX_ROOT="$HPX_ROOT" -DAutoTuneTMP_ROOT=${AutoTuneTMP_ROOT} -DCMAKE_BUILD_TYPE=RelWithDebInfo ../

# uses more than 4G with 4 threads (4G limit on Circle CI)
make -j${PARALLEL_BUILD} VERBOSE=1
cd ../
