

./scripts/slurm/slurm_submit.sh python3 -m src.train_registration --dataset brain-mri --loss l2 --ncc_win_size 9 --deepsim_weights ./weights/brain-mri/segmentation/weights.ckpt --lam 0.1 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=150000