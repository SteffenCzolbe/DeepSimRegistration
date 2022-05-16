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
    # deepsim_0
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim_0/$LAM/ --loss deepsim_0 --deepsim_weights ./weights/platelet-em/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim_1
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim_1/$LAM/ --loss deepsim_1 --deepsim_weights ./weights/platelet-em/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim_2
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim_2/$LAM/ --loss deepsim_2 --deepsim_weights ./weights/platelet-em/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim_01
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim_01/$LAM/ --loss deepsim_01 --deepsim_weights ./weights/platelet-em/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim_02
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim_02/$LAM/ --loss deepsim_02 --deepsim_weights ./weights/platelet-em/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim_12
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim_12/$LAM/ --loss deepsim_12 --deepsim_weights ./weights/platelet-em/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim-ae_0
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim-ae_0/$LAM/ --loss deepsim-ae_0 --deepsim-ae_weights ./weights/platelet-em/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim-ae_1
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim-ae_1/$LAM/ --loss deepsim-ae_1 --deepsim-ae_weights ./weights/platelet-em/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim-ae_2
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim-ae_2/$LAM/ --loss deepsim-ae_2 --deepsim-ae_weights ./weights/platelet-em/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim-ae_01
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim-ae_01/$LAM/ --loss deepsim-ae_01 --deepsim-ae_weights ./weights/platelet-em/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim-ae_02
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim-ae_02/$LAM/ --loss deepsim-ae_02 --deepsim-ae_weights ./weights/platelet-em/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim-ae_12
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim-ae_12/$LAM/ --loss deepsim-ae_12 --deepsim-ae_weights ./weights/platelet-em/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim-transfer
    for LAM in 0.125 0.25 0.5 1 2 4
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim-transfer/$LAM/ --loss deepsim-transfer --deepsim-ae_weights ./weights/phc-u373/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000 
    done

    # deepsim-transfer
    for LAM in 0.125 0.25 0.5 1 2 4
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim-transfer/$LAM/ --loss deepsim-transfer --deepsim_weights ./weights/phc-u373/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000 
    done

    # deepsim-ae-transfer
    for LAM in 0.125 0.25 0.5 1 2 4
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim-transfer-ae/$LAM/ --loss deepsim-transfer-ae --deepsim_weights ./weights/phc-u373/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000 
    done

    # deepsim-vgg
    for LAM in 0.25 0.5 1 2 4
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim-vgg/$LAM/ --loss vgg --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_steps=12500
    done

    # deepsim, extract before warp
    for LAM in 0.0625 0.125 0.25 0.5 1
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim-ebw/$LAM/ --loss deepsim-ebw --deepsim_weights ./weights/platelet-em/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_steps=12500
    done

    # deepsim-ae, extract before warp
    for LAM in 0.0625 0.125 0.25 0.5 1
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/platelet-em/registration/deepsim-ae-ebw//$LAM/ --loss deepsim-ae-ebw --deepsim_weights ./weights/platelet-em/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done
fi

if $TUNE_PHC; then
    # deepsim_0
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim_0/$LAM/ --loss deepsim_0 --deepsim_weights ./weights/phc-u373/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim_1
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim_1/$LAM/ --loss deepsim_1 --deepsim_weights ./weights/phc-u373/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim_2
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim_2/$LAM/ --loss deepsim_2 --deepsim_weights ./weights/phc-u373/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim_01
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim_01/$LAM/ --loss deepsim_01 --deepsim_weights ./weights/phc-u373/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim_02
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim_02/$LAM/ --loss deepsim_02 --deepsim_weights ./weights/phc-u373/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim_12
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim_12/$LAM/ --loss deepsim_12 --deepsim_weights ./weights/phc-u373/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim-ae_0
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim-ae_0/$LAM/ --loss deepsim-ae_0 --deepsim-ae_weights ./weights/phc-u373/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim-ae_1
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim-ae_1/$LAM/ --loss deepsim-ae_1 --deepsim-ae_weights ./weights/phc-u373/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim-ae_2
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim-ae_2/$LAM/ --loss deepsim-ae_2 --deepsim-ae_weights ./weights/phc-u373/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim-ae_01
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim-ae_01/$LAM/ --loss deepsim-ae_01 --deepsim-ae_weights ./weights/phc-u373/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim-ae_02
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim-ae_02/$LAM/ --loss deepsim-ae_02 --deepsim-ae_weights ./weights/phc-u373/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim-ae_12
    for LAM in 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim-ae_12/$LAM/ --loss deepsim-ae_12 --deepsim-ae_weights ./weights/phc-u373/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000
    done

    # deepsim-transfer
    for LAM in 0.125 0.25 0.5 1 2 4
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim-transfer//$LAM/ --loss deepsim-transfer --deepsim_weights ./weights/platelet-em/segmentation/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000 
    done

    # deepsim-ae-transfer
    for LAM in 0.125 0.25 0.5 1 2 4
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim-transfer-ae/$LAM/ --loss deepsim-transfer-ae --deepsim_weights ./weights/platelet-em/autoencoder/weights.ckpt --lam $LAM --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=3000 
    done

    # deepsim-vgg
    for LAM in 0.03125 0.0625 0.125 0.25 0.5
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim-vgg/$LAM/ --loss vgg --lam $LAM --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_steps=10000
    done

    # deepsim, extract before warp
    for LAM in 0.0078125 0.015625 0.03125 0.0625 0.125 0.25
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim-ebw/$LAM/ --loss deepsim-ebw --deepsim_weights ./weights/phc-u373/segmentation/weights.ckpt --lam $LAM --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_steps=10000
    done

    # deepsim-ae, extract before warp
    for LAM in 0.0078125 0.015625 0.03125 0.0625 0.125 0.25
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset phc-u373 --savedir ./out/phc-u373/registration/deepsim-ae-ebw/$LAM/ --loss deepsim-ae-ebw --deepsim_weights ./weights/phc-u373/autoencoder/weights.ckpt --lam $LAM --channels 64 128 256 --batch_size 5 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 2 --distributed_backend ddp --max_steps=10000
    done
fi

if $TUNE_BRAIN; then
    # deepsim, extract before warp
    for LAM in 0.125 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset brain-mri --loss deepsim-ebw --deepsim_weights ./weights/brain-mri/segmentation/weights.ckpt --lam $LAM --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_steps=15000
    done

    # deepsim-ae, extract before warp
    for LAM in 20.125 0.25 0.5 1 2
    do
    $WRAPPER_FUNC python3 -m src.train_registration --net voxelmorph --dataset brain-mri --loss deepsim-ae-ebw --deepsim_weights ./weights/brain-mri/autoencoder/weights.ckpt --lam $LAM --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_steps=15000
    done
fi