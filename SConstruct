import os

vars = Variables("custom.py")
vars.Add("HPX_LIBPATH", "path to the HPX libraries", '/home/pfandedd/git/hpx/build/lib')
vars.Add("HPX_CPPPATH", "path to the HPX includes", ['/home/pfandedd/git/hpx/', '/home/pfandedd/git/hpx/build'])
vars.Add("BOOST_CPPPATH", "path to the boost header files (can be empty)", '')
vars.Add("BOOST_LIBPATH", "path to the boost libs (can be empty)", '')

env = Environment(variables=vars, ENV=os.environ)
env.AppendUnique(CPPPATH=env['HPX_CPPPATH'])
env.AppendUnique(CPPPATH=env['BOOST_CPPPATH'])
env.AppendUnique(LIBS=['hpx', 'hpx_init', 'hpx_iostreams', 'boost_program_options', 'boost_system', 'boost_thread'])
env.PrependUnique(LIBPATH=[env['HPX_LIBPATH'], env['BOOST_LIBPATH']])
env.AppendUnique(CPPFLAGS='-std=c++14')

sources = env.Glob("*.cpp")
objects = [env.Object(s) for s in sources]
env.Program('matrixMultiplicationComponent', objects)
