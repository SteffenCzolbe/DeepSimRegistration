import pickle
import os
import numpy as np
from tqdm import tqdm
from tensorboard.backend.event_processing import event_accumulator
import yaml
from collections import defaultdict
import glob
from .config2D import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
#import matplotlib as mpl
#mpl.rcParams.update(mpl.rcParamsDefault)
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.preview'] = True

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
        #print(hparams)
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
    # set up sup-plots
    fig = plt.figure(figsize=(9, 3))
    axs = fig.subplots(1, len(DATASET_ORDER) + 1, gridspec_kw={'width_ratios': [1, 1, 0.5]})
    axs[2].axis("off")
    #axs = fig.subplots(1, len(DATASET_ORDER))
    plt.subplots_adjust(bottom=0.18, wspace=0.3)

    for i, dataset in enumerate(DATASET_ORDER):
        if dataset not in hparam_tuning_results.keys():
            continue
        for lossfun in EXTRACT_TRANSMORPH_LOSS_FUNCTIONS:
            if lossfun not in hparam_tuning_results[dataset].keys():
                continue
            # read lam, score
            items = hparam_tuning_results[dataset][lossfun].items()
            items = sorted(items, key=lambda t: t[0])
            lambdas, val_dice_overlap = list(zip(*items))
            handle = axs[i].plot(lambdas, val_dice_overlap, color=LOSS_FUNTION_CONFIG[lossfun]
                                 ["primary_color"], label=LOSS_FUNTION_CONFIG[lossfun]["display_name"],
                                                    #linestyle='--' if '_' in lossfun else '-')
                                                    marker=LOSS_FUNTION_CONFIG[lossfun]["marker"],
                                                    markersize=4 if '_' in lossfun else 6,
                                                    linewidth=1 if '_' in lossfun else 1.5)
            handle = handle[0]
            axs[i].set_xscale('log', basex=2)
            axs[i].set_title(PLOT_CONFIG[dataset]["display_name"], fontsize=18)
            LOSS_FUNTION_CONFIG[lossfun]["handle"] = handle
            print(LOSS_FUNTION_CONFIG[lossfun]["handle"])

    # add labels
    fig.text(0.42, 0.04, "Regularization Hyperparameter $\lambda$",
             ha="center", va="center", fontsize=16)
    fig.text(0.05, 0.5, "Val. Mean Dice Overlap", ha="center",
             va="center", rotation="vertical", fontsize=16)

    # add legend
    handles = [
        LOSS_FUNTION_CONFIG[loss_function]["handle"] for loss_function in EXTRACT_TRANSMORPH_LOSS_FUNCTIONS
    ]
    labels = [
        LOSS_FUNTION_CONFIG[loss_function]["display_name"]
        for loss_function in EXTRACT_TRANSMORPH_LOSS_FUNCTIONS
    ]
    
    axs[2].legend(handles, labels, bbox_to_anchor=(1., 1.), fontsize=12)
    #axs[2].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12)
    #axs[2].legend(bbox_to_anchor=(1.04,1), loc="upper left")


    # configure axis precision
    for ax in axs:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    os.makedirs("./out/plots/", exist_ok=True)
    plt.savefig("./out/plots/tranmorph.pdf", bbox_inches="tight")
    plt.savefig("./out/plots/tranmorph.png", bbox_inches="tight")


if __name__ == '__main__':
    hparam_tuning_results = defaultdict(lambda: defaultdict(lambda: {}))

    tuning = glob.glob('./weights_exp/transmorph-platelet/*')
    tuning2 = glob.glob('./weights_exp/transmorph-phc/*')
    runs = tuning + tuning2
    for run in tqdm(runs, desc='reading hparam training logs...'):
        dataset, lossfun, lam = read_model_hparams(run)
        print(run, dataset, lossfun, lam)
        if lossfun in EXTRACT_TRANSMORPH_LOSS_FUNCTIONS:
            mean_val_dice_overlap = read_model_logs(run)
            hparam_tuning_results[dataset][lossfun][lam] = mean_val_dice_overlap

    plot(hparam_tuning_results)
