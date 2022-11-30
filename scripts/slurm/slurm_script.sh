#!/bin/bash -l

# usage:
# submit a slurm-job via
# $ sbatch slurm_script.sh <script to run> <arg1> <arg2> ...

# set job name
#SBATCH --job-name='unnamed job'

# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=2 --mem=30000M

# we run on the gpu partition and we allocate some titanx gpu
#SBATCH -p gpu --gres=gpu:titanrtx:1

#Runtime
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=5-00:00:00

# output files
#SBATCH --output=slurm_logs/slurm-%j.%x.out
#SBATCH --error=slurm_logs/slurm-%j.%x.err

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo Host: 
hostname
echo
echo CUDA_VISIBLE_DEVICES:
echo $CUDA_VISIBLE_DEVICES
echo
echo running command:
echo $@
echo
$@