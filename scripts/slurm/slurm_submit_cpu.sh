#!/bin/bash -l

# usage:
# submit a slurm-job via
# $ slurm_submit.sh <script to run> <arg1> <arg2> ...

# args given to sbatch here will overwrite defaults set in the slurm script
sbatch ./scripts/slurm/slurm_script_cpu.sh $@