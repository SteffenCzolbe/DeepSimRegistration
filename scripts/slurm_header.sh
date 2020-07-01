#!/bin/bash -l

# set job name
#SBATCH --job-name='surm header script'

# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=10000M

# we run on the gpu partition and we allocate some titanx gpu
#SBATCH -p gpu --gres=gpu:titanx:2

#Runtime
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=0-12:00:0
$1