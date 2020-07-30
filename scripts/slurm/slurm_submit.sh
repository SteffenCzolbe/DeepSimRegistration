#!/bin/bash -l

# usage:
# submit a slurm-job via
# $ slurm_submit.sh <script to run> <arg1> <arg2> ...

# comment given to sbatch here will overwrite defaults set in the slurm script
sbatch --job-name='submitted via script' --time=0-00:01:00 ./scripts/slurm/slurm_script.sh $@