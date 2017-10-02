#!/bin/bash
set -x
set -e

if [ -z ${matrix_multiplication_source_me_sourced} ] ; then
    source source-me.sh
fi

if [ ! -d "AutoTuneTMP" ]; then
    git clone git@github.com:DavidPfander-UniStuttgart/AutoTuneTMP.git
    cd AutoTuneTMP
    git submodule init
    git submodule update
    cd ..
fi

cd AutoTuneTMP
git pull
cd ..
# else
#     cd hpx
#     git pull
#     cd ..
# fi

mkdir -p AutoTuneTMP/build
cd AutoTuneTMP/build

echo "compiling AutoTuneTMP"
# detection of Vc doesn't work with a relative path
# > cmake_AutoTuneTMP.log 2>&1
cmake -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DVc_ROOT="$Vc_ROOT" -DCMAKE_BUILD_TYPE=release ../

# uses more than 4G with 4 threads (4G limit on Circle CI)
#   > make_AutoTuneTMP.log 2>&1
make -j${PARALLEL_BUILD} VERBOSE=1
cd ../..
