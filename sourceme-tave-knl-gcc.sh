# module unload gcc/4.9.3

module load craype-mic-knl
module switch PrgEnv-cray/6.0.3 PrgEnv-gnu
module load CMake/3.8.1

# module list
export CRAYPE_LINK_TYPE=dynamic
# export BOOST_ROOT=/users/pfandedd/scratch/tave/mic-knl-gcc-build/boost_1_63
# export basedir=/users/pfandedd/scratch/tave
# #export myarch=${CRAY_CPU_TARGET}
# export myarch=mic-knl-gcc
# export hpxtoolchain=${basedir}/src/hpx/cmake/toolchains/CrayKNL.cmake
# export buildtype=Release
# # export malloc=jemalloc
# export malloc=jemalloc

# special flags for some library builds
export mycflags="-fPIC -march=knl -ffast-math"
export mycxxflags="-fPIC -march=knl -ffast-math"
export myldflags="-fPIC"
export mycc=gcc
export mycxx=g++
export myfc=gfortran

source source-me.sh

