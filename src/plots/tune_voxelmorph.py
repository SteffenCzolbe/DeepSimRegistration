from collections import defaultdict
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os
import pickle
from tqdm import tqdm
from tensorboard.backend.event_processing import event_accumulator
import yaml

from .config import *


def read_tb_scalar_log(file, scalar):
    """
    reads the tensorboard log.
    """
    ea = event_accumulator.EventAccumulator(
        file,
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        },
    )
    ea.Reload()  # loads events from file
    events = ea.Scalars(scalar)

    values = []
    for event in events:
        values.append(event.value)

    return values

def read_hparams_from_yaml(file):
    with open(file) as f:
        hparams = yaml.load(f, Loader=yaml.Loader)
    return hparams['dataset'], hparams['loss'], hparams['lam']

def read_model_hparams(dir):
    dataset, lossfun, lam = read_hparams_from_yaml(
        os.path.join(dir, 'hparams.yaml'))
    return dataset, lossfun, lam

def read_model_logs(dir):
    files = os.listdir(dir)
    log_files = list(
        map(lambda f: os.path.join(dir, f), filter(
            lambda s: "events.out" in s, files))
    )
    mean_val_dice_overlap = max(
        read_tb_scalar_log(log_files[0], 'val/dice_overlap'))

    return mean_val_dice_overlap

def plot(hparam_tuning_results):
    fig = plt.figure(figsize=(8.2, 6.0)) # (width, height)
    axs = fig.subplots(2, 2)
    axs = axs.flatten()
    plt.subplots_adjust(bottom=0.11, left=0.18, wspace=0.3, hspace=0.3, right=0.9)


    for i, dataset in enumerate(DATASET_ORDER):
        if dataset not in hparam_tuning_results.keys():
            continue
        for lossfun in MIND_AND_OTHER_LOSS_FUNTION:
            if lossfun not in hparam_tuning_results[dataset].keys():
                continue
            # read lam, score
            items = hparam_tuning_results[dataset][lossfun].items()
            items = sorted(items, key=lambda t: t[0])
            lambdas, val_dice_overlap = list(zip(*items))
            handle = axs[i].plot(lambdas, val_dice_overlap, color=LOSS_FUNTION_CONFIG[lossfun]
                                 ["primary_color"], label=LOSS_FUNTION_CONFIG[lossfun]["display_name"], 
                                                    marker=LOSS_FUNTION_CONFIG[lossfun]["marker"],
                                                    markersize=4 if '_' in lossfun else 6,
                                                    linewidth=1 if '_' in lossfun else 1.5)
            axs[i].set_xscale('log', basex=2)
            axs[i].set_title(PLOT_CONFIG[dataset]["display_name"], fontsize=18)
    
    # add labels
    fig.text(0.5, 0.05, "Regularization Hyperparameter $\lambda$",
             ha="center", va="center", fontsize=16)
    fig.text(0.07, 0.5, "Val. Mean Dice Overlap", ha="center",
             va="center", rotation="vertical", fontsize=16)


    # configure axis precision
    for ax in axs:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    os.makedirs("./out/plots/pdf/", exist_ok=True)
    os.makedirs("./out/plots/png/", exist_ok=True)
    plt.savefig("./out/plots/pdf/voxelmorph.pdf", bbox_inches="tight")
    plt.savefig("./out/plots/png/voxelmorph.png", bbox_inches="tight")


if __name__ == '__main__':
    hparam_tuning_results = defaultdict(lambda: defaultdict(lambda: {}))

    baselines = glob.glob('./weights/hparam_tuning/*')
    deepsim = glob.glob('./weights_experiments/voxelmorph/deep-sim/*')
    mind_2d = glob.glob('./weights_experiments/voxelmorph/mind/mind-voxelmorph-2d/*')
    mind_3d_brain = glob.glob('./weights_experiments/voxelmorph/mind/mind-voxelmorph-3d-brain/*')

    runs = mind_2d + mind_3d_brain + baselines + deepsim 
    for run in tqdm(runs, desc='reading hparam training logs...'):
        dataset, lossfun, lam = read_model_hparams(run)
        #print(run, dataset, lossfun, lam)
        if lossfun in MIND_AND_OTHER_LOSS_FUNTION:
            mean_val_dice_overlap = read_model_logs(run)
            hparam_tuning_results[dataset][lossfun][lam] = mean_val_dice_overlap

    plot(hparam_tuning_results)
