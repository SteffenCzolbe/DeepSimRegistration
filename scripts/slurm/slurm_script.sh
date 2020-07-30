#!/bin/bash -l

# usage:
# submit a slurm-job via
# $ sbatch ./scripts/slurm/slurm_script.sh <script to run>

# set job name
#SBATCH --job-name='my job'

# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=30000M

# we run on the gpu partition and we allocate some titanx gpu
#SBATCH -p gpu --gres=gpu:titanrtx:1

#Runtime
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=5-00:00:00

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo Host: 
hostname
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
$1