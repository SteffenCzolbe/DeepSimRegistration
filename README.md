# Semantic similarity metrics for learned image registration

Steffen Czolbe, Oswin Krause, Aasa Feragen

[[MIDL 2021 open-review]](https://openreview.net/forum?id=9M5cH--UdcC) [[Med-NeurIPS 2020 Short-Paper]](https://arxiv.org/abs/2011.05735) [[Med-NeurIPS 2020 Oral]](https://youtu.be/GV4r2fOe0Oo)

[![Video](https://img.youtube.com/vi/GV4r2fOe0Oo/hqdefault.jpg)](https://youtu.be/GV4r2fOe0Oo)

This repository contains all experiments presented in the paper, the code used to generate the figures, and instructions and scripts to re-produce all results. Implementation in the deep-learning framework pytorch.

# Publications

## DeepSim: Semantic similarity metrics for learned image registration

NeurIPS workshop on Medical Imaging, 2020. Oral presentation (top 10% of submissions). Included initial concept and use of a segmentation model for feature extraction. Cite as:

```
@article{czolbe2020deepsim,
title={DeepSim: Semantic similarity metrics for learned image registration},
author={Czolbe, Steffen and Krause, Oswin and Feragen, Aasa},
journal={NeurIPS workshop on Medical Imaging},
year={2020}
}
```

## Semantic similarity metrics for learned image registration

Under review for MIDL 2021 (single-blind review). Expands on the initial publication with the use auf autoencoders for unsupervised training of the feature extractor, an expanded evaluation section, and a more detailed description of the work.

# Reproduce Experiments

This section gives a short overview of how to reproduce the experiments presented in the paper. Most steps have a script present in the `scripts/` subdirectory. If started locally, they will start training on local accessible GPUs. If started in a cluster-environment administered by the task scheduling system Slurm, training runs are scheduled as slurm-jobs instead.

Training logs are written into a directory `lightning_logs/<run no.>`. Each run contains checkpoint files, event logs tracking various training and validation metrics, and a `hparams.yaml` file containing all the hyperparameters of the run. All logs can be easily monitored via `tensorboard --logdir lightning_logs/`.

## Dependencies

All dependencies are listed the the file `requirements.txt`. Simply install them with a package manager of your choice, eg.

```
pip3 install -r requirements.txt
```

## Data

Since we are not allowed to re-distribute the datasets, it is required to perform manual action. 

### Brain-MRI
The Brain-MRI scans have been taken from the publically accessible ABIDEI, ABIDEII, OASIS3 studies. We used Freesurfer and some custom scripts to perform the preprocessing steps of
- intensity normalization
- skullstripping
- affine alignment
- automatic segmentation
- crop to 160x192x224
- Segmentation areas of LH and RH combined to single labels
- some smaller segmentations removed/combined, Total 22 classes left

The resulting intensity and label volumes are then organized in a separate directory:

```
deepsimreg/
    <you are here>
brain_mris/
    data/
        <subject_id>/
            "brain_aligned.nii.gz"
            "seg_coalesced_aligned.nii.gz"
```
### PhC-373
The PhC-373 dataset can be automatically downloaded and pre-processed using the provided script
```
$ scripts/download_phc.sh
```

### Platelet-EM
The Platelet-EM dataset originally contains 5 class annotations. Due to the strong imbalance of labels, we combined some of the less frequent classes to end up with 3 groups (Background, Cytoplasm and Organelle). The datset needs to be placed in:
```
deepsimreg/
    data/
        platelet_em_reduced/
            images/
                24-images.tif
                50-images.tif
            labels-class/
                24-class.tif
                50-class.tif
```

## Train segmentation models for feature extraction

First, we train the segmentation models to obtain weights for the DeepSim metric.

```
$ scripts/train_seg.sh
```

During training, logs and checkpoints will be written to `lightning_logs/<run no.>`. To take advantage of the default-model paths specified as command args, it is necessary to manually copy the weights-file in the `lightning_logs/<run no.>/checkpoints/` subdirectory to the corresponding directory `weights/segmentation/<dataset>/weights.ckpt`. Alternatively, all paths to weights in the future steps will have to be specified explicitly.

The quality of the segmentation models can be quantitatively and qualitatively assessed by performing segmentation of the test set with

```
$ scripts/test_segmentation.sh
```

, which will print metrics on the test set to stdout and create segmented .tif and .nii.gz files in the `out/` directory.

## Train autoencoders for feature extraction

The training of autoencoders follow the same principle as the segmentation models. Train the autoencoders by running:

```
$ scripts/train_autoencoder.sh
```

Then manually copy the trained models from `lightning_logs/<run no.>/checkpoints/` to `weights/autoencoder/<dataset>/weights.ckpt`. Autoencoder quality can be checked with `scripts/test_autoencoder.sh`.

## Train Registration Models

Next, we train the registration models. All models with pre-tuned hyperparameters can be trained by executing

```
$ scrits/train_registration.sh
```

Optionally, hyperparameters can be tuned by running

```
$ scrits/tune_registration.sh
```

Equivalent to the segmentation part, trained weights have to be manually extracted from the logs to take advantage of the default paths set by commands. Copy the weights-file in the `lightning_logs/<run no.>/checkpoints/` subdirectory to the corresponding directory `weights/registration/<dataset>/<loss function>/weights.ckpt`.

The quality of the registration can

The quality of the registration models can be quantitatively and qualitatively assessed by performing registration of the test set with

```
$ scrits/test_registration.sh
```

, which will print metrics on the test set to stdout and create segmented .tif and .nii.gz files in the `out/` directory.

## Plots

The Plots from the paper can be re-created by running

```
$ scripts/plot.sh
```

The plots are written to `out/plots/`.
