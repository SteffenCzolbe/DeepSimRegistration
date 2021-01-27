# Semantic similarity metrics for learned image registration

Steffen Czolbe, Oswin Krause, Aasa Feragen

[[Med-NeurIPS 2020 Short-Paper]](https://arxiv.org/abs/2011.05735) [[Med-NeurIPS 2020 Oral]](https://youtu.be/GV4r2fOe0Oo)

[![Video](https://img.youtube.com/vi/GV4r2fOe0Oo/hqdefault.jpg)](https://youtu.be/GV4r2fOe0Oo)

This repository contains all experiments presented in the paper, and instructions and code to re-run the experiments. Implementation in the deep-learning framework PyTorch.

## Dependencies

All dependencies are listed the the file `requirements.txt`. Simply install them by

```
pip3 install -r requirements.txt
```

## Data

Preprocessed versions of the PhC-373 and Platelet-EM datasets in enclosed in the repository. The Brain-MRI scans are not provided with this package. By default they have to be placed in a separate repository, structured as:

```
deepsimreg/
    data/
        PhC-U373/
            images/
                01.tif
                02.tif
            labels-class/
                01.tif
                02.tif
        platelet_em_reduced/
            images/
                24-images.tif
                50-images.tif
            labels-class/
                24-class.tif
                50-class.tif
brain_mris/
    data/
        <subject_id>/
            "brain_aligned.nii.gz"
            "seg_coalesced_aligned.nii.gz"
```

# Reproduce Experiments

This section gives a short overview of how to reproduce the experiments presented in the paper. Most steps have a script present in the `scripts/` subdirectory. If started locally, they will start training on local accessible GPUs. If started in a cluster-environment administered by the task scheduling system Slurm, training runs are scheduled as slurm-jobs instead.

Training logs are written into a directory `lightning_logs/<run no.>`. Each run contains checkpoint files, event logs tracking various training and validation metrics, and a `hparams.yaml` file containing all the hyperparameters necessary to re-create the run. All logs can be easily monitored via `tensorboard --logdir lightning_logs/`.

## Train Feature Extractors

First, we train the segmentation models to obtain weights for the DeepSim metric.

```
$ scrits/train_seg.sh
```

During training, logs and checkpoints will be written to `lightning_logs/<run no.>`. To take advantage of the default-model paths specified as command args, it is necessary to manually copy the weights-file in the `lightning_logs/<run no.>/checkpoints/` subdirectory to the corresponding directory `weights/segmentation/<dataset>/weights.ckpt`. Alternatively, all paths to weights in the future steps will have to be specified explicitly.

The quality of the segmentation models can be quantitatively and qualitatively assessed by performing segmentation of the test set with

```
$ scrits/test_segmentation.sh
```

, which will print metrics on the test set to stdout and create segmented .tif and .nii.gz files in the `out/` directory.

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
