#!/bin/bash
set -x
set -e

if [[ ! $PARALLEL_BUILD ]]; then
    echo "boost: PARALLEL_BUILD not set, defaulting to 2"
    export PARALLEL_BUILD=2
fi

if [ ! -d "boost_1_65_0/" ]; then
    wget 'http://downloads.sourceforge.net/project/boost/boost/1.65.0/boost_1_65_0.tar.bz2'
    tar xf boost_1_65_0.tar.bz2

    # configure for gcc 7
    echo "using gcc : 7.1 : /usr/bin/g++-7  ; " > boost_1_65_0/tools/build/src/user-config.jam
fi

if [ ! -d "boost_1_65_0_install/" ]; then
    echo "building boost"
    cd boost_1_65_0
    ./bootstrap.sh --prefix="$Boost_ROOT" > bootstrap_boost.log 2>&1
    #-d2: more verbose output
    ./b2 -j${PARALLEL_BUILD} cxxflags="$CXX_FLAGS" variant=release -d1 install > b2_boost.log 2>&1
    cd ..
fi



