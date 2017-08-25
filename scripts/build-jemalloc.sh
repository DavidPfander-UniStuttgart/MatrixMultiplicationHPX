#!/bin/bash
set -x
# set -e not set, because jemalloc doc does not build successfully


if [ ! -d jemalloc ] ; then
    git clone https://github.com/jemalloc/jemalloc.git
    cd jemalloc
    git checkout 4.5.0
    cd ..
fi

cd jemalloc
# export CC=${mycc}
# export CXX=${mycxx}
# export CFLAGS=${mycflags}
# export CXXFLAGS=${mycxxflags}
# make clean

autoconf
if [ ! -d build ] ; then
    mkdir build
fi
cd build
echo "building jemalloc"
../configure CC=cc CXX=CC --prefix=${JEMALLOC_ROOT} > configure_jemalloc.log 2>&1
make -j${PARALLEL_BUILD} > make_jemalloc.log 2>&1
make install  > make_install_jemalloc.log 2>&1
cd ../..
