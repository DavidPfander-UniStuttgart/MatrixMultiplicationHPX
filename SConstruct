env = Environment()
env.Replace(CC=['gcc-6'])
env.Replace(CXX=['g++-6'])
env.AppendUnique(CPPPATH=['/home/pfandedd/git/hpx/', '/home/pfandedd/git/hpx/build'])
env.AppendUnique(LIBS=['hpx', 'hpx_init', 'hpx_iostreams', 'boost_program_options', 'boost_system', 'boost_thread'])
env.AppendUnique(LIBPATH=['/home/pfandedd/git/hpx/build/lib'])

sources = env.Glob("*.cpp")
objects = [env.Object(s) for s in sources]
env.Program('matrixMultiplicationComponent', objects)