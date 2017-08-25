#!/bin/bash -e
set -x
set -e

if [ ! -d "Vc/" ]; then
    git clone https://github.com/STEllAR-GROUP/Vc.git
    cd Vc
    git checkout pfandedd_inlining_AVX512
    git checkout HEAD~1
    cd ..
fi
# else
#     cd Vc
#     git pull
#     cd ..
# fi

mkdir -p Vc/build
cd Vc/build
cmake -DCMAKE_INSTALL_PREFIX="$Vc_ROOT" -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=release ../
make -j${PARALLEL_BUILD} VERBOSE=1 install
cd ../..
