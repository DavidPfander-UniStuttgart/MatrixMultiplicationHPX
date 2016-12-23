import os

vars = Variables("custom.py")
vars.Add("PKG_CONFIG_PATH_RELEASE", "path to the pkg-config for configuring hpx, used for release builds", '')
vars.Add("PKG_CONFIG_PATH_DEBUG", "path to the pkg-config for configuring hpx, used for debug builds", '')
vars.Add("BOOST_ROOT_RELEASE", "Boost root path, release", '')
vars.Add("BOOST_ROOT_DEBUG", "Boost root path, debug", '')
vars.Add("VC_INCLUDE_PATH", "include files of the Vc vectorization library (not needed for HPX with datapar)", '')
vars.Add("VC_LIBRARY_PATH", "path to shared libararies of the Vc vectorization library (not needed for HPX with datapar)", '')
vars.Add("CC", "C compiler", 'gcc')
vars.Add("CXX", "C++ compiler", 'g++')

env_release = Environment(variables=vars, ENV=os.environ)
env_release.AppendUnique(CPPFLAGS=['-fopenmp', '-Wall', '-Wno-ignored-attributes', '-Wno-unused-local-typedefs'])
env_release.AppendUnique(LINKFLAGS=['-fopenmp'])
env_release.AppendUnique(CPPPATH=[env_release["VC_INCLUDE_PATH"]])
env_release.AppendUnique(LIBPATH=[env_release["VC_LIBRARY_PATH"]])
env_release.AppendUnique(LIBS=['Vc'])


env_debug = env_release.Clone()

env_release['ENV']['PKG_CONFIG_PATH']= env_release['PKG_CONFIG_PATH_RELEASE']
env_release.ParseConfig('pkg-config --cflags --libs hpx_application')
env_release.AppendUnique(LIBS=['hpx_iostreams'])
env_release.AppendUnique(CPPFLAGS=['-O3', '-ffast-math', '-mavx2', '-mfma', '-g'])
env_release.AppendUnique(LIBPATH=[env_release['BOOST_ROOT_RELEASE'] + "/lib"])
env_release.AppendUnique(CPPPATH=[env_release['BOOST_ROOT_RELEASE'] + "/include"])

env_debug['ENV']['PKG_CONFIG_PATH']= env_debug['PKG_CONFIG_PATH_DEBUG']
env_debug.ParseConfig('pkg-config --cflags --libs hpx_application_debug')
env_debug.AppendUnique(LIBS=['hpx_iostreamsd'])
env_debug.AppendUnique(CPPFLAGS=['-O0', '-g', '-fno-omit-frame-pointer'])
env_debug.AppendUnique(LIBPATH=[env_debug['BOOST_ROOT_DEBUG'] + "/lib"])
env_debug.AppendUnique(CPPPATH=[env_debug['BOOST_ROOT_DEBUG'] + "/include"])
env_debug.AppendUnique(CPPDEFINES=["DEBUG"])

SConscript('src/SConscript', variant_dir='release', exports={'env':env_release}, duplicate=0)
# debug is a bad idea with Vc, as the vector would need to be actually portable
# SConscript('src/SConscript', variant_dir='debug', exports={'env':env_debug}, duplicate=0)
