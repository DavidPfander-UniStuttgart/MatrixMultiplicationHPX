#!/bin/bash
#SBATCH --job-name="XeonE3MatMultTuner"
#SBATCH --nodes=1

export PATH=/home/pfandedd/SC17_Poster_sgscl/AutoTuneTMP/gcc_install/bin:$PATH
srun ./build_RelWithDebInfo/combined_tuning xeon_e3

