#!/bin/bash

# trains the models.

# Check if slurm compute cluster available. Submit as slurm job if possible.
if sbatch -h &> /dev/null; then
    echo "Submitting to slurm..."
    WRAPPER_FUNC=scripts/slurm/slurm_submit.sh
else
    echo "Running locally..."
    WRAPPER_FUNC=
fi

$WRAPPER_FUNC python3 -m src.train_segmentation --dataset platelet-em --max_epochs 50000 --savedir ./out/platelet-em/segmentation/ --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp
$WRAPPER_FUNC python3 -m src.train_segmentation --dataset phc-u373 --max_epochs 50000 --savedir ./out/phc-u373/segmentation/ --channels 64 128 256 --batch_size 5 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp
$WRAPPER_FUNC python3 -m src.train_segmentation --dataset brain-mri --max_steps 100000 --channels 32 64 128 --batch_size 1 --accumulate_grad_batches 4 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp
