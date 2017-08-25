#!/bin/bash
set -x
set -e

if [[ ! $PARALLEL_BUILD ]]; then
    echo "hpx: PARALLEL_BUILD not set, defaulting to 1"
    export PARALLEL_BUILD=1
fi

if [ ! -d "hpx/" ]; then
    git clone https://github.com/STEllAR-GROUP/hpx.git
    # cd hpx
    # git checkout 1.0.0
    # cd ..
fi
# else
#     cd hpx
#     git pull
#     cd ..
# fi

mkdir -p hpx/build
cd hpx/build

echo "building hpx"

# detection of Vc doesn't work with a relative path
cmake -DBOOST_ROOT="$Boost_ROOT" -DHPX_WITH_CXX14=ON -DHPX_WITH_EXAMPLES=OFF -DHPX_WITH_DATAPAR_VC=ON -DHPX_WITH_DATAPAR_VC_NO_LIBRARY=ON -DVc_ROOT="$Vc_ROOT" -DCMAKE_INSTALL_PREFIX="$HPX_ROOT" -DJEMALLOC_ROOT=${JEMALLOC_ROOT} -DHPX_WITH_MALLOC=jemalloc -DCMAKE_CXX_STANDARD_LIBRARIES="-latomic" -DCMAKE_BUILD_TYPE=release ../ > cmake_hpx.log 2>&1

# uses more than 4G with 4 threads (4G limit on Circle CI)
# not using a logfile as Circle CI complains about no output for 10 minutes
#  > make_install_hpx.log 2>&1
# VERBOSE=1
make -j${PARALLEL_BUILD}  install
cd ../..
