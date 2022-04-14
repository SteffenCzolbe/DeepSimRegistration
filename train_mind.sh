python3 -m src.train_registration --net voxelmorph --dataset brain-mri --loss mind --lam 8 --channels 32 64 128 --batch_size 1 --accumulate_grad_batches 4 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=30
python3 -m src.train_registration --net voxelmorph --dataset brain-mri --loss mind --lam 4 --channels 32 64 128 --batch_size 1 --accumulate_grad_batches 4 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=30
python3 -m src.train_registration --net voxelmorph --dataset brain-mri --loss mind --lam 2 --channels 32 64 128 --batch_size 1 --accumulate_grad_batches 4 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=30
python3 -m src.train_registration --net voxelmorph --dataset brain-mri --loss mind --lam 1 --channels 32 64 128 --batch_size 1 --accumulate_grad_batches 4 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=30
python3 -m src.train_registration --net voxelmorph --dataset brain-mri --loss mind --lam 0.5 --channels 32 64 128 --batch_size 1 --accumulate_grad_batches 4 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=30
python3 -m src.train_registration --net voxelmorph --dataset brain-mri --loss mind --lam 0.25 --channels 32 64 128 --batch_size 1 --accumulate_grad_batches 4 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=30
python3 -m src.train_registration --net voxelmorph --dataset brain-mri --loss mind --lam 0.125 --channels 32 64 128 --batch_size 1 --accumulate_grad_batches 4 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=30
python3 -m src.train_registration --net voxelmorph --dataset brain-mri --loss mind --lam 0.0625 --channels 32 64 128 --batch_size 1 --accumulate_grad_batches 4 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=30
python3 -m src.train_registration --net voxelmorph --dataset brain-mri --loss mind --lam 0.03125 --channels 32 64 128 --batch_size 1 --accumulate_grad_batches 4 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=30
python3 -m src.train_registration --net voxelmorph --dataset brain-mri --loss mind --lam 0.015625 --channels 32 64 128 --batch_size 1 --accumulate_grad_batches 4 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=30

# brain-mri: scrips/train_registration.sh
#python3 -m src.train_registration --dataset brain-mri --loss l2 --lam 0.08 --channels 32 64 128 --batch_size 1 --gpus -1 --lr 0.0001 --bnorm --dropout --accumulate_grad_batches 4 --max_epochs=30

# test mind on 2D dataset
#python3 -m src.train_registration --net voxelmorph --dataset platelet-em --savedir ./out/test/mind/ --loss mind --lam 1 --accumulate_grad_batches 2 --channels 64 128 256 --batch_size 3 --gpus -1 --lr 0.0001 --bnorm --dropout --distributed_backend ddp --max_epochs=10
