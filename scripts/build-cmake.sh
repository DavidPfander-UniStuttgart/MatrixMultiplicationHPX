#!/bin/bash
set -x
set -e

if [ -z ${matrix_multiplication_source_me_sourced} ] ; then
    source source-me.sh
fi

if [ ! -d cmake ] ; then
    wget https://cmake.org/files/v3.9/cmake-3.9.3-Linux-x86_64.tar.gz
    tar xf cmake-3.9.3-Linux-x86_64.tar.gz
    mv cmake-3.9.3-Linux-x86_64 cmake
fi
