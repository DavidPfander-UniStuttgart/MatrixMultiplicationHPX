#!/bin/bash
set -x
set -e

if [ -z ${matrix_multiplication_source_me_sourced} ] ; then
    source source-me.sh
fi

if [ ! -d "cppjit" ]; then
    git clone git@github.com:DavidPfander-UniStuttgart/cppjit.git
    cd cppjit
    git submodule init
    git submodule update
    cd ..
fi

cd cppjit
git pull
cd ..
# else
#     cd hpx
#     git pull
#     cd ..
# fi

mkdir -p cppjit/build
cd cppjit/build

echo "compiling cppjit"
# detection of Vc doesn't work with a relative path
# > cmake_cppjit.log 2>&1
cmake -DCMAKE_INSTALL_PREFIX="$CPPJIT_ROOT" -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_BUILD_TYPE=release ../

# uses more than 4G with 4 threads (4G limit on Circle CI)
#   > make_cppjit.log 2>&1
make -j${PARALLEL_BUILD} VERBOSE=1
make VERBOSE=1 install
cd ../..
