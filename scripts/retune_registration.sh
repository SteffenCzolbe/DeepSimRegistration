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
    # l2
    for LAM in 0.04
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/l2/$LAM/ --loss l2 --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_steps=12500
    done
fi

if $TUNE_PHC; then
    # l2
    for LAM in  0.000078125 0.00015625 0.0003125 0.000625
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/l2/$LAM/ --loss l2 --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_steps=10000
    done

    # ncc2
    for LAM in 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/ncc2/$LAM/ --loss ncc2 --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_steps=10000
    done

    # deepsim
    for LAM in 1
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim/$LAM/ --loss deepsim --deepsim_weights ./weights/phc-u373/segmentation/weights.ckpt --lam $LAM --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_steps=10000
    done
fi

if $TUNE_BRAIN; then
    # l2
    for LAM in 0.02 0.32
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss l2 --ncc_win_size 9 --lam $LAM --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_steps=15000
    done

    # ncc2
    for LAM in 0.25 4
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss ncc2 --ncc_win_size 9 --lam $LAM --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_steps=15000
    done

    # ncc2+supervised
    for LAM in 0.5 8
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss ncc2+supervised --ncc_win_size 9 --lam $LAM --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_steps=15000
    done

    # deepsim
    for LAM in 0.5 8
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss deepsim --ncc_win_size 9 --deepsim_weights ./weights/brain-mri/segmentation/weights.ckpt --lam $LAM --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_steps=15000
    done
fi