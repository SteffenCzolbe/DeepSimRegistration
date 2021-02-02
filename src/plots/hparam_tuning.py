import pickle
import os
import numpy as np
from tqdm import tqdm
from .config import *
from tensorboard.backend.event_processing import event_accumulator
import yaml
from collections import defaultdict
import glob
from .config import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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


def read_model(dir):
    files = os.listdir(dir)
    log_files = list(
        map(lambda f: os.path.join(dir, f), filter(lambda s: "events.out" in s, files))
    )
    mean_val_dice_overlap = max(read_tb_scalar_log(log_files[0], 'val/dice_overlap'))
    
    dataset, lossfun, lam = read_hparams_from_yaml(os.path.join(dir, 'hparams.yaml'))
    
    return dataset, lossfun, lam, mean_val_dice_overlap

def plot(hparam_tuning_results):
    # set up sup-plots
    fig = plt.figure(figsize=(8.5, 2.5))
    axs = fig.subplots(1, len(DATASET_ORDER))
    plt.subplots_adjust(bottom=0.15)
    
    for i, dataset in enumerate(DATASET_ORDER):
        if dataset not in hparam_tuning_results.keys():
            continue
        for lossfun in LOSS_FUNTION_ORDER:
            if lossfun not in hparam_tuning_results[dataset].keys():
                continue
            # read lam, score
            items = hparam_tuning_results[dataset][lossfun].items()
            items = sorted(items,key=lambda t: t[0])
            lambdas, val_dice_overlap = list(zip(*items))
            axs[i].plot(lambdas, val_dice_overlap, color=LOSS_FUNTION_CONFIG[lossfun]["primary_color"], label=LOSS_FUNTION_CONFIG[lossfun]["display_name"])
            axs[i].scatter(lambdas, val_dice_overlap, color=LOSS_FUNTION_CONFIG[lossfun]["primary_color"], marker='x')
            axs[i].set_xscale('log', basex=2)
            axs[i].set_title(PLOT_CONFIG[dataset]["display_name"])
        
    # add labels
    fig.text(0.5, 0.015, "Regularization Hyperparameter $\lambda$", ha="center", va="center")
    fig.text(0.07, 0.5, "Validation Mean Dice Overlap", ha="center", va="center", rotation="vertical")
    #axs[-1].legend()
    # configure precision
    for ax in axs:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    os.makedirs("./out/plots/", exist_ok=True)
    plt.savefig("./out/plots/hparam.pdf")
    plt.savefig("./out/plots/hparam.png")
            

if __name__ == '__main__':
    hparam_tuning_results = defaultdict(lambda: defaultdict(lambda: {}))
    
    runs = glob.glob('./weights/hparam_tuning/*')
    for run in tqdm(runs, desc='reading hparam training logs...'):
        dataset, lossfun, lam, mean_val_dice_overlap = read_model(run)
        hparam_tuning_results[dataset][lossfun][lam] = mean_val_dice_overlap
    
    plot(hparam_tuning_results)
    