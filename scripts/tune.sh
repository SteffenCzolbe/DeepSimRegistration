#!/bin/bash

# hyperparameter tuning.
TUNE_PLATELET=true
TUNE_PHC=true
TUNE_BRAIN=true


# Check if slurm compute cluster available. Submit as slurm job if possible.
if sbatch -h &> /dev/null; then
    echo "Submitting to slurm..."
    WRAPPER_FUNC=scripts/slurm/slurm_submit.sh
else
    echo "Running locally..."
    WRAPPER_FUNC=
fi

if $TUNE_PLATELET; then
    # l2
    for LAM in 0.005 0.01 0.02 0.04 0.08
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/l2/$LAM/ --loss l2 --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # ncc
    for LAM in 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/ncc/$LAM/ --loss ncc --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # ncc+supervised
    for LAM in 0.25 0.5 1
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/ncc+supervised/$LAM/ --loss ncc+supervised --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 3 --accumulate_grad_batches 2 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim
    for LAM in 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim/$LAM/ --loss deepsim --deepsim_weights ./weights/platelet-em/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim transfer
    for LAM in 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim_transfer/$LAM/ --loss deepsim-transfer --deepsim_weights ./weights/phc-u373/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # vgg
    for LAM in 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset platelet-em --savedir ./out/platelet-em/registration/vgg/$LAM/ --loss vgg --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done
fi

if $TUNE_PHC; then
    # l2
    for LAM in 0.005 0.01 0.02 0.04 0.08
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/l2/$LAM/ --loss l2 --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
    done

    # ncc
    for LAM in 0.25 0.5 1
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/ncc/$LAM/ --loss ncc --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
    done

    # ncc+supervised
    for LAM in 0.125 0.25 0.5
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/ncc+supervised/$LAM/ --loss ncc+supervised --ncc_win_size 9 --lam $LAM --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
    done

    # deepsim
    for LAM in 0.125 0.5 1
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim/$LAM/ --loss deepsim --deepsim_weights ./weights/phc-u373/segmentation/weights.ckpt --lam $LAM --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
    done

    # deepsim transfer
    for LAM in 0.06175 0.125 0.5
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim_transfer/$LAM/ --loss deepsim-transfer --deepsim_weights ./weights/platelet-em/segmentation/weights.ckpt --lam $LAM --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
    done

    # vgg
    for LAM in 0.125 0.5 1
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset phc-u373 --savedir ./out/phc-u373/registration/vgg/$LAM/ --loss vgg --lam $LAM --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_epochs=3000
    done
fi

if $TUNE_BRAIN; then
    # l2
    for LAM in 0.01 0.02 0.04
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss l2 --ncc_win_size 9 --lam $LAM --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=150000
    done

    # ncc
    for LAM in 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss ncc --ncc_win_size 9 --lam $LAM --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=150000
    done

    # ncc+supervised
    for LAM in 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss ncc+supervised --ncc_win_size 9 --lam $LAM --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=150000
    done

    # deepsim
    for LAM in 0.25 0.5 1
    do
    $WRAPPER_FUNC python3 -m src.train_registration --dataset brain-mri --loss deepsim --ncc_win_size 9 --deepsim_weights ./weights/brain-mri/segmentation/weights.ckpt --lam $LAM --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=150000
    done
fi