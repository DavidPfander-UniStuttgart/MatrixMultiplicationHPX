#!/bin/bash
set -x
set -e

if [ -z ${matrix_multiplication_source_me_sourced} ] ; then
    source source-me.sh
fi

if [ ! -d "boost" ]; then
    wget 'http://downloads.sourceforge.net/project/boost/boost/1.65.0/boost_1_65_0.tar.bz2'
    tar xf boost_1_65_0.tar.bz2
    mv boost_1_65_0 boost

    # configure for gcc 7
    if [[ "$MATRIX_MULTIPLICATION_TARGET" != "knl" ]]; then
	echo "using gcc : 7.1 : /usr/bin/g++-7  ; " > boost/tools/build/src/user-config.jam
    fi
fi

if [ ! -d "boost_install/" ]; then
    echo "building boost"
    cd boost
    ./bootstrap.sh --prefix="$BOOST_ROOT" > bootstrap_boost.log 2>&1
    #-d2: more verbose output
    # not using a logfile as Circle CI complains about no output for 10 minutes
    # > b2_boost.log 2>&1
    ./b2 -j${PARALLEL_BUILD} cxxflags="$CXX_FLAGS" variant=release -d1 install
    cd ..
fi



