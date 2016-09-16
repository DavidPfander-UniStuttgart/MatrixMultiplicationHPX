import os

vars = Variables("custom.py")
vars.Add("PKG_CONFIG_PATH_RELEASE", "path to the pkg-config for configuring hpx, used for release builds", '')
vars.Add("PKG_CONFIG_PATH_DEBUG", "path to the pkg-config for configuring hpx, used for debug builds", '')

env_release = Environment(variables=vars, ENV=os.environ)
env_debug = env_release.Clone()

env_release['ENV']['PKG_CONFIG_PATH']= env_release['PKG_CONFIG_PATH_RELEASE']
env_release.ParseConfig('pkg-config --cflags --libs hpx_application')
env_release.AppendUnique(LIBS=['hpx_iostreams'])
env_release.AppendUnique(CPPFLAGS=['-O3', '-fopenmp'])
env_release.AppendUnique(LINKFLAGS=['-fopenmp'])

env_debug['ENV']['PKG_CONFIG_PATH']= env_debug['PKG_CONFIG_PATH_DEBUG']
env_debug.ParseConfig('pkg-config --cflags --libs hpx_application_debug')
env_debug.AppendUnique(LIBS=['hpx_iostreamsd'])
env_debug.AppendUnique(CPPFLAGS=['-O0', '-g'])

SConscript('SConscript', variant_dir='release', exports={'env':env_release})
SConscript('SConscript', variant_dir='debug', exports={'env':env_debug})
