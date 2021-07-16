#!/bin/bash

# hyperparameter tuning.
TUNE_PLATELET=true
TUNE_PHC=true
TUNE_BRAIN=true


# Check if slurm compute cluster available. Submit as slurm job if possible.
if sbatch -h &> /dev/null; then
    echo "Submitting to slurm..."
    WRAPPER_FUNC=scripts/slurm/slurm_submit_nice.sh
else
    echo "Running locally..."
    WRAPPER_FUNC=
fi

if $TUNE_PLATELET; then


    # nmi, specific search
    for LAM in 0.125 0.25 0.5 1 2 4
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/nmi/$LAM/ --loss nmi --lam $LAM --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_steps=12500
    done
fi

if $TUNE_PHC; then
    # nmi
    for LAM in 0.125 0.25 0.5 1 2 4
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/nmi/$LAM/ --loss nmi --lam $LAM --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_steps=10000
    done
fi

if $TUNE_BRAIN; then
    # nmi
    for LAM in 0.125 0.25 0.5 1 2 4
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss nmi --lam $LAM --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_steps=15000
    done
fi