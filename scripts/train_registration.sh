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

# pletelet
$WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/l2/ --loss l2 --lam 0.02 --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/ncc2/ --loss ncc2 --ncc_win_size 9 --lam 0.5 --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/ncc2+supervised/ --loss ncc2+supervised --ncc_win_size 9 --lam 0.5 --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/nmi/ --loss nmi --lam 0.25 --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/mind/ --loss mind --lam 0.015625 --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim/ --loss deepsim --deepsim_weights ./weights/platelet-em/segmentation/weights.ckpt --lam 1 --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim-vgg/ --loss vgg --lam 1 --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim-ae/ --loss deepsim-ae --deepsim_weights ./weights/platelet-em/autoencoder/weights.ckpt --lam 1 --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000

# PhC
$WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/l2/ --loss l2 --lam 0.005 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/ncc2/ --loss ncc2 --ncc_win_size 9 --lam 0.25 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/ncc2+supervised/ --loss ncc2+supervised --ncc_win_size 9 --lam 0.25 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/nmi/ --loss nmi --lam 0.5 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/mind/ --loss mind --lam 0.03125 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim/ --loss deepsim --deepsim_weights ./weights/phc-u373/segmentation/weights.ckpt --lam 0.125 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim-vgg/ --loss vgg --lam 0.125 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
$WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim-ae/ --loss deepsim-ae --deepsim_weights ./weights/phc-u373/autoencoder/weights.ckpt --lam 0.125 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000

# brains
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss l2 --lam 0.08 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=30
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss ncc2 --ncc_win_size 9 --lam 1.0 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=30
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss ncc2+supervised --ncc_win_size 9 --lam 2.0 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=30
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss nmi --lam 2 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=30
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss mind --lam 2 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=30
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss deepsim --deepsim_weights ./weights/brain-mri/segmentation/weights.ckpt --lam 2.0 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=30
$WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss deepsim-ae --deepsim_weights ./weights/brain-mri/autoencoder/weights.ckpt --lam 1.0 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=30

# hippocampusmr
$WRAPPER_FUNC python3 -m src.train_registration --dataset hippocampusmr --loss l2 --lam 0.08 --channels 32 64 128 --batch_size 4 --gpus -1 --lr 0.0001 --max_steps=1000000
$WRAPPER_FUNC python3 -m src.train_registration --dataset hippocampusmr --loss ncc2 --lam 1 --channels 32 64 128 --batch_size 4 --gpus -1 --lr 0.0001 --max_steps=1000000
$WRAPPER_FUNC python3 -m src.train_registration --dataset hippocampusmr --loss ncc2+supervised --lam 1 --ncc_win_size 9 --channels 32 64 128 --batch_size 4 --gpus -1 --lr 0.0001 --max_steps=1000000
$WRAPPER_FUNC python3 -m src.train_registration --dataset hippocampusmr --loss nmi --lam 1 --ncc_win_size 9 --channels 32 64 128 --batch_size 4 --gpus -1 --lr 0.0001 --max_steps=1000000
$WRAPPER_FUNC python3 -m src.train_registration --dataset hippocampusmr --loss mind --lam 0.25 --channels 32 64 128 --batch_size 4 --gpus -1 --lr 0.0001 --max_steps=1000000
$WRAPPER_FUNC python3 -m src.train_registration --dataset hippocampusmr --loss deepsim --lam 0.5 --deepsim_weights ./weights/hippocampusmr/segmentation/weights.ckpt --channels 32 64 128 --batch_size 4 --gpus -1 --lr 0.0001 --max_steps=1000000
$WRAPPER_FUNC python3 -m src.train_registration --dataset hippocampusmr --loss deepsim-ae --lam 0.25 --deepsim_weights ./weights/hippocampusmr/autoencoder/weights.ckpt --channels 32 64 128 --batch_size 4 --gpus -1 --lr 0.0001 --max_steps=1000000