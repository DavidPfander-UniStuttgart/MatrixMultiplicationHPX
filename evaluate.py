#!/usr/bin/python3.4

import shlex, subprocess, os, re
import matplotlib.pyplot as plt

def run(n, repetition, transposed, blockInput):
    cmd = 'release/matrix_multiply --n-value=' + str(n) + ' --repetitions=' + str(repetition) + \
        ' --small-block-size=128 --verbose=0 --check=false --algorithm=pseudodynamic --hpx:cores=' + \
        str(cores) + ' --hpx:thread=' + str(threads) + ' --max-work-difference=0 --transposed=' + \
        str(transposed) + ' --block-input=' + str(blockInput)
    # cmd = r'bash test.sh'
    # cmd = r'ldd release/matrix_multiply'
    args = shlex.split(cmd)
    print args
    p = subprocess.Popen(args, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = p.communicate()

    # print '----stdout----'
    # print stdout
    totalTime = re.search(r'total time: (.*?)s', stdout).group(1)
    averageTimePerRun = re.search(r'average time per run: (.*?)s', stdout).group(1)
    performance = re.search(r'performance: (.*?)Gflops', stdout).group(1)

    print 'totalTime:', totalTime
    print 'averageTimePerRun:', averageTimePerRun
    print 'performance:', performance
    return performance

hpxReleaseLibs = '/home/pfandedd/git/hpx/build/lib'

if 'LD_LIBRARY_PATH' in os.environ:
    print('LD_LIBRARY_PATH ->' + str(os.environ['LD_LIBRARY_PATH']))
    os.environ['LD_LIBRARY_PATH'] += ':' + hpxReleaseLibs
else:
    os.environ['LD_LIBRARY_PATH'] = ':' + hpxReleaseLibs

# print('LD_LIBRARY_PATH ->' + str(os.environ['LD_LIBRARY_PATH']))

n_list = [512, 1024, 2048, 4096, 8192]
repetition_list = [10, 10, 5, 2, 1]
cores = 16
threads = 16

performance_blocked_list = []

for n, repetition in zip(n_list, repetition_list):
    performance = run(n, repetition, True, 128)
    performance_blocked_list.append(performance)

plt.plot(n_list, performance_blocked_list, label='blocked')

performance_unblocked_list = []
for n, repetition in zip(n_list, repetition_list):
    performance = run(n, repetition, True, 0)
    performance_unblocked_list.append(performance)

plt.plot(n_list, performance_unblocked_list, label='non-blocked')

plt.legend(loc=2)
plt.show()

# # pointsToPlot = []
# for n, repetition, performance in zip(n_list, repetition_list, performance_blocked_list):
#     print 'n:', n, ' ->', performance, ('(rep:' + str(repetition) + ')')
#     # pointsToPlot += [n, performance]
