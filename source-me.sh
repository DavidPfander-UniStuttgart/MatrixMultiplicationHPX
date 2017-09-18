

export CC=gcc-7
export CXX=g++-7
if [[ "$MATRIX_MULTIPLICATION_TARGET" = "knl" ]]; then
    export CXX_FLAGS="-march=knl -mtune=knl"
else
    export CXX_FLAGS="-march=native -mtune=native"
fi
export Vc_ROOT=$PWD/Vc_install
export Boost_ROOT=$PWD/boost_1_65_0_install
export BOOST_ROOT=$Boost_ROOT
export HPX_ROOT=$PWD/hpx_install
# not installed!
export AutoTuneTMP_ROOT=$PWD/AutoTuneTMP
export MatrixMultiplicationHPX_ROOT=$PWD/MatrixMultiplicationHPX_install
export JEMALLOC_ROOT=$PWD/jemalloc_install

export LD_LIBRARY_PATH=$PWD/boost_1_65_0_install/lib:$PWD/hpx_install/lib:$PWD/Vc_install/lib
