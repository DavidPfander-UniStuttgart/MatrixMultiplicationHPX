#!/bin/bash -e
set -x

if [ ! -d "hpx/" ]; then
    git clone https://github.com/STEllAR-GROUP/hpx.git
    cd hpx
    git checkout 1.0.0
    cd ..
fi
# else
#     cd hpx
#     git pull
#     cd ..
# fi

mkdir -p hpx/build
cd hpx/build

# detection of Vc doesn't work with a relative path
cmake -DBOOST_ROOT="$Boost_ROOT" -DHPX_WITH_EXAMPLES=OFF -DHPX_WITH_DATAPAR_VC=ON -DHPX_WITH_DATAPAR_VC_NO_LIBRARY=ON -DVc_ROOT="$Vc_ROOT" -DCMAKE_INSTALL_PREFIX="$HPX_ROOT" -DJEMALLOC_ROOT=${JEMALLOC_ROOT} -DHPX_WITH_MALLOC=jemalloc -DCMAKE_BUILD_TYPE=release ../

# uses more than 4G with 4 threads (4G limit on Circle CI)
make -j${PARALLEL_BUILD} VERBOSE=1 install
cd ../..
