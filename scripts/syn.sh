#!/bin/bash

# Check if slurm compute cluster available. Submit as slurm job if possible.
if sbatch -h &> /dev/null; then
    echo "Submitting to slurm..."
    WRAPPER_FUNC=scripts/slurm/slurm_submit_cpu.sh
else
    echo "Running locally..."
    WRAPPER_FUNC=
fi

# brain-mri
$WRAPPER_FUNC python3 -m src.syn --dataset brain-mri --feature_extractor none --data_from 0.0 --data_to 0.5 --out_file ./out/brain-mri/syn/syn.yaml
$WRAPPER_FUNC python3 -m src.syn --dataset brain-mri --feature_extractor none --data_from 0.5 --data_to 1.0 --out_file ./out/brain-mri/syn/syn.yaml
$WRAPPER_FUNC python3 -m src.syn --dataset brain-mri --feature_extractor ae --feature_extractor_weights ./weights/brain-mri/autoencoder/weights.ckpt --data_from 0 --data_to 0.5 --out_file ./out/brain-mri/syn/syn_ae.yaml
$WRAPPER_FUNC python3 -m src.syn --dataset brain-mri --feature_extractor ae --feature_extractor_weights ./weights/brain-mri/autoencoder/weights.ckpt --data_from 0.5 --data_to 1.0 --out_file ./out/brain-mri/syn/syn_ae.yaml
$WRAPPER_FUNC python3 -m src.syn --dataset brain-mri --feature_extractor seg --feature_extractor_weights ./weights/brain-mri/segmentation/weights.ckpt --data_from 0 --data_to 0.5 --out_file ./out/brain-mri/syn/syn_seg.yaml
$WRAPPER_FUNC python3 -m src.syn --dataset brain-mri --feature_extractor seg --feature_extractor_weights ./weights/brain-mri/segmentation/weights.ckpt --data_from 0.5 --data_to 1.0 --out_file ./out/brain-mri/syn/syn_seg.yaml

# platelet-em
$WRAPPER_FUNC python3 -m src.syn --dataset platelet-em --feature_extractor none --out_file ./out/platelet-em/syn/syn.yaml
$WRAPPER_FUNC python3 -m src.syn --dataset platelet-em --feature_extractor ae --feature_extractor_weights ./weights/platelet-em/autoencoder/weights.ckpt --out_file ./out/platelet-em/syn/syn_ae.yaml
$WRAPPER_FUNC python3 -m src.syn --dataset platelet-em --feature_extractor seg --feature_extractor_weights ./weights/platelet-em/segmentation/weights.ckpt --out_file ./out/platelet-em/syn/syn_seg.yaml

# phc-u373
$WRAPPER_FUNC python3 -m src.syn --dataset phc-u373 --feature_extractor none --out_file ./out/phc-u373/syn/syn.yaml
$WRAPPER_FUNC python3 -m src.syn --dataset phc-u373 --feature_extractor ae --feature_extractor_weights ./weights/phc-u373/autoencoder/weights.ckpt --out_file ./out/phc-u373/syn/syn_ae.yaml
$WRAPPER_FUNC python3 -m src.syn --dataset phc-u373 --feature_extractor seg --feature_extractor_weights ./weights/phc-u373/segmentation/weights.ckpt --out_file ./out/phc-u373/syn/syn_seg.yaml
