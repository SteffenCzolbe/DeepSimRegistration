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

for LAM in 0.25 0.5 1
do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/ncc2/$LAM/ --loss ncc2 --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    $WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/ncc2+supervised/$LAM/ --loss ncc2+supervised --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
done

for LAM in 0.25 0.5 1
do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/ncc2/$LAM/ --loss ncc2 --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
    $WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/ncc2+supervised/$LAM/ --loss ncc2+supervised --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
done

$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss l2 --lam 0.08 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=150000
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss ncc2 --ncc_win_size 9 --lam 1.0 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=150000
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss ncc2+supervised --ncc_win_size 9 --lam 2.0 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=150000
