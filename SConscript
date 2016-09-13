Import('env')

sources = env.Glob("*.cpp")
objects = [env.Object(s) for s in sources]
env.Program('matrix_multiply', objects)
