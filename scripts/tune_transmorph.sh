#!/bin/bash

# hyperparameter tuning.
TUNE_PLATELET=true
TUNE_PHC=true

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
    for LAM in 0.005 0.01 0.02 0.04 0.08
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net transmorph2d --dataset platelet-em --savedir ./out/platelet-em/transmorph/l2/$LAM/ --loss l2 --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # ncc2
    for LAM in 0.125 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net transmorph2d --dataset platelet-em --savedir ./out/platelet-em/transmorph/ncc2/$LAM/ --loss ncc2 --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # ncc2+supervised
    for LAM in 0.125 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net transmorph2d --dataset platelet-em --savedir ./out/platelet-em/transmorph/ncc2+supervised/$LAM/ --loss ncc2+supervised --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # nmi
    for LAM in  0.0625 0.125 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net transmorph2d --dataset platelet-em --savedir ./out/platelet-em/transmorph/nmi/$LAM/ --loss nmi --lam $LAM --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # mind
    for LAM in 0.03125 0.0625 0.125 0.25 0.5 1
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net transmorph2d --dataset platelet-em --savedir ./out/platelet-em/transmorph/mind/$LAM/ --loss mind --lam $LAM --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim
    for LAM in 0.25 0.5 1 2 4
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net transmorph2d --dataset platelet-em --savedir ./out/platelet-em/transmorph/deepsim/$LAM/ --loss deepsim --deepsim_weights ./weights/platelet-em/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim-ae
    for LAM in 0.25 0.5 1 2 4
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net transmorph2d --dataset platelet-em --savedir ./out/platelet-em/transmorph/deepsim-ae/$LAM/ --loss deepsim-ae --deepsim_weights ./weights/platelet-em/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done
fi

if $TUNE_PHC; then
    # l2
    for LAM in  0.000078125 0.00015625 0.0003125 0.000625 0.00125  0.0025 0.005 0.01 0.02
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net transmorph2d --dataset phc-u373 --savedir ./out/phc-u373/transmorph/l2/$LAM/ --loss l2 --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
    done

    # ncc2
    for LAM in 0.0625 0.125 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net transmorph2d --dataset phc-u373 --savedir ./out/phc-u373/transmorph/ncc2/$LAM/ --loss ncc2 --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
    done

    # ncc2+supervised
    for LAM in 0.0625 0.125 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net transmorph2d --dataset phc-u373 --savedir ./out/phc-u373/transmorph/ncc2+supervised/$LAM/ --loss ncc2+supervised --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
    done

    # nmi
    for LAM in  0.0625 0.125 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net transmorph2d --dataset phc-u373 --savedir ./out/phc-u373/transmorph/nmi/$LAM/ --loss nmi --lam $LAM --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
    done

    # mind
    for LAM in  0.0078125 0.015625 0.03125 0.0625 0.125 0.25
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net transmorph2d --dataset phc-u373 --savedir ./out/phc-u373/transmorph/mind/$LAM/ --loss mind --lam $LAM --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
    done

    # deepsim
    for LAM in 0.03125 0.0625 0.125 0.25 0.5 1
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net transmorph2d --dataset phc-u373 --savedir ./out/phc-u373/transmorph/deepsim/$LAM/ --loss deepsim --deepsim_weights ./weights/phc-u373/segmentation/weights.ckpt --lam $LAM --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
    done

    # deepsim-ae
    for LAM in 0.03125 0.0625 0.125 0.25 0.5 1
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net transmorph2d --dataset phc-u373 --savedir ./out/phc-u373/transmorph/deepsim-ae/$LAM/ --loss deepsim-ae --deepsim_weights ./weights/phc-u373/autoencoder/weights.ckpt --lam $LAM --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
    done

fi
