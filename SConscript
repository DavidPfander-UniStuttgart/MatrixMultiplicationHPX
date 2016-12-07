Import('env')

sources = env.Glob("*.cpp")
sources += env.Glob("reference_kernels/*.cpp")
sources += env.Glob("variants/*.cpp")
objects = [env.Object(s) for s in sources]
env.Program('matrix_multiply', objects)
