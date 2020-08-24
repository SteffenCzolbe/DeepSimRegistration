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

$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --load_from_checkpoint ./weights/brain-mri/registration/l2/weights.ckpt --loss l2 --lam 0.08 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=150000
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --load_from_checkpoint ./weights/brain-mri/registration/ncc/weights.ckpt --loss ncc --ncc_win_size 9 --lam 1.0 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=150000
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --load_from_checkpoint ./weights/brain-mri/registration/ncc+supervised/weights.ckpt --loss ncc+supervised --ncc_win_size 9 --lam 2.0 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=150000
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --load_from_checkpoint ./weights/brain-mri/registration/deepsim/weights.ckpt --loss deepsim --deepsim_weights ./weights/brain-mri/segmentation/weights.ckpt --lam 2.0 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=150000
