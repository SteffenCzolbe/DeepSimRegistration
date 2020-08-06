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

$WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/l2/ --loss l2 --lam 0.02 --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/ncc/ --loss ncc --ncc_win_size 9 --lam 0.5 --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/ncc+supervised/ --loss ncc+supervised --ncc_win_size 9 --lam 0.5 --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim/ --loss deepsim --deepsim_weights ./weights/platelet-em/segmentation/weights.ckpt --lam 1 --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim_transfer/ --loss deepsim-transfer --deepsim_weights ./weights/phc-u373/segmentation/weights.ckpt --lam 0.5 --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/vgg/ --loss vgg --lam 1 --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000

$WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/l2/ --loss l2 --lam 0.005 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/ncc/ --loss ncc --ncc_win_size 9 --lam 0.5 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/ncc+supervised/ --loss ncc+supervised --ncc_win_size 9 --lam 0.5 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim/ --loss deepsim --deepsim_weights ./weights/phc-u373/segmentation/weights.ckpt --lam 0.125 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim_transfer/ --loss deepsim-transfer --deepsim_weights ./weights/platelet-em/segmentation/weights.ckpt --lam 0.06125 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/vgg/ --loss vgg --lam 0.125 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000

# TODO: brain-mri
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss l2 --lam 0.01 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=150000
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss ncc --ncc_win_size 9 --lam 0.5 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=150000
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss ncc+supervised --ncc_win_size 9 --lam 0.5 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=150000
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss deepsim --deepsim_weights ./weights/brain-mri/segmentation/weights.ckpt --lam 0.25 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=150000
