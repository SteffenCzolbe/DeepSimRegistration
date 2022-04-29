""" create a scatterplot of dice overlap vs transformation smoothness.
    """
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import pickle
import numpy as np
from .config import *
import os
from matplotlib.ticker import FormatStrFormatter
from .run_models import run_models

def load_data_for_model(dataset, loss_function):
    # load data
    # with open(args.cache_file_name, 'rb') as f:
    #     test_results = pickle.load(f)
    test_results = run_models(use_cached=True)
    if dataset not in test_results.keys():
        return None, None
    if loss_function not in test_results[dataset].keys():
        return None, None
    dice = test_results[dataset][loss_function]["dice_overlap"].mean(axis=0)
    log_var = test_results[dataset][loss_function]["jacobian_determinant_log_var"]
    #smoothness = test_results[dataset][loss_function]["jacobian_determinant_log_var"].mean(axis=0)
    smoothness = log_var[~np.isnan(log_var)].mean(axis=0)
    folding = test_results[dataset][loss_function]["jacobian_determinant_negative"].mean(axis=0)
    return dice, smoothness, folding


def main(args):

    # set up sup-plots
    # fig = plt.figure(figsize=(10, 3))
    # axs = fig.subplots(1, len(DATASET_ORDER))
    # plt.subplots_adjust(bottom=0.33)

    fig = plt.figure(figsize=(8, 3))
    axs = fig.subplots(1, len(DATASET_ORDER))
    plt.subplots_adjust(bottom=0.33, wspace=0.275)


    for i, dataset in enumerate(DATASET_ORDER):
        for loss_function in LOSS_FUNTION_ORDER:
            dice, smoothness, folding = load_data_for_model(dataset, loss_function)
            print(dataset, loss_function, np.round(dice, 2), np.round(smoothness,2), np.round(folding * 100, 2))
            # if dataset =='platelet-em' and loss_function =='mind':
            #     smoothness = 2.427
            if dice is None:
                continue
            # read lam, score
            handle = axs[i].scatter(
                dice, smoothness, color=LOSS_FUNTION_CONFIG[loss_function]["primary_color"], marker=LOSS_FUNTION_CONFIG[loss_function]["marker"], s=80)
            axs[i].set_title(PLOT_CONFIG[dataset]["display_name"], fontsize=18)
            LOSS_FUNTION_CONFIG[loss_function]["handle"] = handle

    # add labels
    fig.text(0.5, 0.2, "Test Mean Dice Overlap",
             ha="center", va="center", fontsize=16)
    fig.text(
        0.04, 0.58, "$\sigma^2(\log |J_{\Phi}|)$", ha="center", va="center", rotation="vertical", fontsize=16)

    # add legend
    handles = [
        LOSS_FUNTION_CONFIG[loss_function]["handle"] for loss_function in LOSS_FUNTION_ORDER
    ]
    labels = [
        LOSS_FUNTION_CONFIG[loss_function]["display_name"]
        for loss_function in LOSS_FUNTION_ORDER
    ]
    fig.legend(handles, labels, loc="lower center",
               ncol=len(handles), handlelength=1.5, columnspacing=1.5)

    # configure axis precision
    for i, ax in enumerate(axs):
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if i == 2:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        else:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #ax.set_ylim(ymin=0, ymax=0.6)

    os.makedirs("./out/plots/pdf/", exist_ok=True)
    os.makedirs("./out/plots/png/", exist_ok=True)
    plt.savefig("./out/plots/pdf/smoothness_vs_dice_overlap.pdf", bbox_inches='tight')
    plt.savefig("./out/plots/png/smoothness_vs_dice_overlap.png", bbox_inches='tight')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--cache_file_name', type=str, default='./src/plots/cache.pickl', help='File with test results.')
    args = parser.parse_args()
    main(args)
