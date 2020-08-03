


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
The Plots from the paper can be created by running
```
$ scripts/plot.sh
```
The plots are written to `out/plots/`.