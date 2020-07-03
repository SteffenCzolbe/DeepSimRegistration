#!/bin/bash -l

# usage:
# submit a slurm-job via
# $ sbatch ./scripts/slurm_header.sh <your script to run>

# set job name
#SBATCH --job-name='reg phc'

# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=30000M

# we run on the gpu partition and we allocate some titanx gpu
#SBATCH -p gpu --gres=gpu:titanrtx:1

#Runtime
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=0-18:00:00

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

hostname
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
srun python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/$1/$2/ --loss $1 --ncc_win_size 9 --deepsim_weights ./weights/phc-u373/segmentation/weights.ckpt --lam $2 --channels 64 128 256 --batch_size 5 --gpus 1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --max_epochs=5000
