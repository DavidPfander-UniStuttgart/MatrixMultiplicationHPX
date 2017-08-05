#!/bin/bash -e
set -x

if [ ! -d "AutoTuneTMP" ]; then
    git clone git@github.com:DavidPfander-UniStuttgart/AutoTuneTMP.git
    cd AutoTuneTMP
    git submodule init
    git submodule update
    cd ..
fi
# else
#     cd hpx
#     git pull
#     cd ..
# fi

# mkdir -p AutoTuneTMP/build
# cd AutoTuneTMP/build

# detection of Vc doesn't work with a relative path
# cmake -DVc_ROOT="$AutoTuneTMP_ROOT" -DCMAKE_BUILD_TYPE=release ../

# uses more than 4G with 4 threads (4G limit on Circle CI)
# make -j8 VERBOSE=1 install
# cd ../..
