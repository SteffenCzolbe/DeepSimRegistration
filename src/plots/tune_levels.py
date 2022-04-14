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
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
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
    return hparams['dataset'], hparams['loss'], hparams['lam'], hparams['deepsim_weights']


def read_model_hparams(dir):
    dataset, lossfun, lam, w = read_hparams_from_yaml(
        os.path.join(dir, 'hparams.yaml'))
    return dataset, lossfun, lam, w


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
        for lossfun in EXTRACT_LEVEL_AE_LOSS_FUNTIONS:
            if lossfun not in hparam_tuning_results[dataset].keys():
                continue
            # read lam, score
            items = hparam_tuning_results[dataset][lossfun].items()
            items = sorted(items, key=lambda t: t[0])
            lambdas, val_dice_overlap = list(zip(*items))
            handle = axs[i].plot(lambdas, val_dice_overlap, color=LOSS_FUNTION_CONFIG[lossfun]
                                 ["primary_color"], label=LOSS_FUNTION_CONFIG[lossfun]["display_name"], 
                                                    linestyle='--' if '_' in lossfun else '-',
                                                    marker='<' if 'transfer' in lossfun else LOSS_FUNTION_CONFIG[lossfun]["marker"],
                                                    #markersize=4 if '_' in lossfun else 6,
                                                    linewidth=1 if '_' in lossfun else 1.5)
            handle = handle[0]
            axs[i].set_xscale('log', basex=2)
            axs[i].set_title(PLOT_CONFIG[dataset]["display_name"], fontsize=18)
            LOSS_FUNTION_CONFIG[lossfun]["handle"] = handle
            #print(LOSS_FUNTION_CONFIG[lossfun]["handle"])

    # add labels
    fig.text(0.42, 0.04, "Regularization Hyperparameter $\lambda$",
             ha="center", va="center", fontsize=16)
    fig.text(0.05, 0.5, "Val. Mean Dice Overlap", ha="center",
             va="center", rotation="vertical", fontsize=16)

    # add legend
    handles = [
        LOSS_FUNTION_CONFIG[loss_function]["handle"] for loss_function in EXTRACT_LEVEL_AE_LOSS_FUNTIONS
    ]
    labels = [
        LOSS_FUNTION_CONFIG[loss_function]["display_name"]
        for loss_function in EXTRACT_LEVEL_AE_LOSS_FUNTIONS
    ]
    axs[2].legend(handles, labels, bbox_to_anchor=(1., 1.), fontsize=12)

    # configure axis precision
    for ax in axs:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    os.makedirs("./out/plots/", exist_ok=True)

    plt.savefig("./out/plots/ae_levels.pdf", bbox_inches="tight")
    plt.savefig("./out/plots/ae_levels.png", bbox_inches="tight")

    # plt.savefig("./out/plots/seg_levels.pdf", bbox_inches="tight")
    # plt.savefig("./out/plots/seg_levels.png", bbox_inches="tight")


if __name__ == '__main__':
    hparam_tuning_results = defaultdict(lambda: defaultdict(lambda: {}))

    platelet_seg = glob.glob('./weights_exp/levels-ae-platelet/lightning_logs/*')
    platelet_ae = glob.glob('./weights_exp/levels-seg-platelet/lightning_logs/*')

    # tuning = glob.glob('./weights_exp/deep-sim/*')
    # phc_seg = glob.glob('./logs_levels_seg/lightning_logs/*')
    # phc_seg2 = glob.glob('./logs_levels_seg2/lightning_logs/*')
    # phc_ae = glob.glob('./logs_levels_ae/lightning_logs/*')
    #runs = phc_seg + phc_seg2 + phc_ae + tuning + platelet_seg + platelet_ae

    tuning = glob.glob('./weights_exp/deep-sim-1/*')
    phc_seg = glob.glob('./weights_exp/levels-seg-phc/*')
    phc_ae = glob.glob('./weights_exp/levels-ae-phc/*')
    runs = phc_seg + phc_ae + tuning + platelet_seg + platelet_ae
    
    for run in tqdm(runs, desc='reading hparam training logs...'):
        dataset, lossfun, lam, w = read_model_hparams(run)
        folder = run.split('/')[-1]
        print(folder, dataset, lossfun, lam, w)
        if lossfun in EXTRACT_LEVEL_AE_LOSS_FUNTIONS:
            if lam != 4:
                mean_val_dice_overlap = read_model_logs(run)
                hparam_tuning_results[dataset][lossfun][lam] = mean_val_dice_overlap

    plot(hparam_tuning_results)
