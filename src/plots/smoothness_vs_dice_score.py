""" create a scatterplot of dice overlap vs transformation smoothness.
    """
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import pickle
import numpy as np
import os
from matplotlib.ticker import FormatStrFormatter
from .run_models import run_models
from .config import *

def load_data_for_model(dataset, loss_function):
    # with open(args.cache_file_name, 'rb') as f:
    #     test_results = pickle.load(f)

    # load data
    test_results = run_models(use_cached=True, model=args.net)
    
    if dataset not in test_results.keys():
        return None, None
    if loss_function not in test_results[dataset].keys():
        return None, None
    dice = test_results[dataset][loss_function]["dice_overlap"].mean(axis=0)
    log_var = test_results[dataset][loss_function]["jacobian_determinant_log_var"]
    smoothness = log_var[~np.isnan(log_var)].mean(axis=0)
    if args.net == 'voxelmorph':
        folding = test_results[dataset][loss_function]["jacobian_determinant_negative"].mean(axis=0)
        return dice, smoothness, folding
    else:
        return dice, smoothness
    

def main(args):
    if args.net == 'voxelmorph':
        fig = plt.figure(figsize=(8, 3))
        axs = fig.subplots(1, len(DATASET_ORDER))
        plt.subplots_adjust(bottom=0.33, wspace=0.275)
    else:
        DATASET_ORDER.remove("brain-mri") 
        fig = plt.figure(figsize=(9, 3))
        axs = fig.subplots(1, len(DATASET_ORDER) + 1, gridspec_kw={'width_ratios': [1, 1, 0.5]})
        axs[2].axis("off")
        plt.subplots_adjust(bottom=0.18, wspace=0.3)




    for i, dataset in enumerate(DATASET_ORDER):
        for loss_function in LOSS_FUNTION_ORDER:
            if args.net == 'voxelmorph':
                dice, smoothness, folding = load_data_for_model(dataset, loss_function)
                print(dataset, loss_function, np.round(dice, 3), np.round(smoothness,2), np.round(folding * 100, 2))
            else:
                dice, smoothness = load_data_for_model(dataset, loss_function)
                print(dataset, loss_function, dice, smoothness)

            if dice is None:
                continue
            # read lam, score
            handle = axs[i].scatter(
                dice, smoothness, color=LOSS_FUNTION_CONFIG[loss_function]["primary_color"], marker=LOSS_FUNTION_CONFIG[loss_function]["marker"], s=80)
            axs[i].set_title(PLOT_CONFIG[dataset]["display_name"], fontsize=18)
            LOSS_FUNTION_CONFIG[loss_function]["handle"] = handle

    # add labels
    if args.net == "voxelmorph":
        x1, y1 = 0.5, 0.2
        x2, y2 = 0.04, 0.58  
    else:
        x1, y1 = 0.42, 0.04
        x2, y2 = 0.05, 0.5
    fig.text(x1, y1, "Test Mean Dice Overlap", ha="center", va="center", fontsize=16)
    fig.text(x2, y2, "$\sigma^2(\log |J_{\Phi}|)$", ha="center", va="center", rotation="vertical", fontsize=16)

    # add legend
    handles = [
        LOSS_FUNTION_CONFIG[loss_function]["handle"] for loss_function in LOSS_FUNTION_ORDER
    ]
    labels = [
        LOSS_FUNTION_CONFIG[loss_function]["display_name"]
        for loss_function in LOSS_FUNTION_ORDER
    ]
    
    # configure axis precision
    if args.net == 'voxelmorph':
        fig.legend(handles, labels, loc="lower center",
               ncol=len(handles), handlelength=1.5, columnspacing=1.5)
               
        for i, ax in enumerate(axs):
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if i == 2:
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            else:
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    else:
        axs[2].legend(handles, labels, bbox_to_anchor=(1.05, 1.), fontsize=12)
        for i,ax in enumerate(axs):
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if i == 1:
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            else:
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[0].locator_params(axis='x', nbins=5)
        axs[1].locator_params(axis='x', nbins=3)


    os.makedirs("./out/plots/pdf/", exist_ok=True)
    os.makedirs("./out/plots/png/", exist_ok=True)
    plt.savefig(f"./out/plots/pdf/smoothness_vs_dice_overlap_{args.net}.pdf", bbox_inches='tight')
    plt.savefig(f"./out/plots/png/smoothness_vs_dice_overlap_{args.net}.png", bbox_inches='tight')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--net', type=str, default='voxelmorph', help='voxelmorph or transmorph.')
    args = parser.parse_args()
    main(args)
