#!/bin/bash
set -x
set -e

if [ -z ${matrix_multiplication_source_me_sourced} ] ; then
    source source-me.sh
fi

if [ ! -d jemalloc ] ; then
    git clone https://github.com/jemalloc/jemalloc.git
    cd jemalloc
    git checkout 4.5.0
    cd ..
fi

cd jemalloc

autoconf
if [ ! -d build ] ; then
    mkdir build
fi
cd build
echo "building jemalloc"
../configure CC=cc CXX=CC --prefix=${JEMALLOC_ROOT} > configure_jemalloc.log 2>&1
make -j${PARALLEL_BUILD} > make_jemalloc.log 2>&1
make install_include install_lib > make_install_jemalloc.log 2>&1
cd ../..
