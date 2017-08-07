#!/bin/bash -e
set -x

if [ ! -d "MatrixMultiplicationHPX" ]; then
    git clone git@github.com:DavidPfander-UniStuttgart/MatrixMultiplicationHPX.git
    cd MatrixMultiplicationHPX
    git checkout autotuning
    cd ..
fi

cd MatrixMultiplicationHPX
git pull
cd ..
# else
#     cd hpx
#     git pull
#     cd ..
# fi

mkdir -p MatrixMultiplicationHPX/build
cd MatrixMultiplicationHPX/build

# detection of Vc doesn't work with a relative path
# cmake -DVc_ROOT="$MatrixMultiplicationHPX_ROOT" -DCMAKE_BUILD_TYPE=release ../
cmake -DHPX_ROOT="$HPX_ROOT" -DAutoTuneTMP_ROOT=${AutoTuneTMP_ROOT} -DCMAKE_BUILD_TYPE=release ../

# uses more than 4G with 4 threads (4G limit on Circle CI)
make -j8 VERBOSE=1
cd ../..
