import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from src.registration_model import RegistrationModel
import torch
from .config import *
from .run_models import run_models
from matplotlib.ticker import FormatStrFormatter

# set up sup-plots
fig = plt.figure(figsize=(7, 2.5))
axs = fig.subplots(1, len(DATASET_ORDER))
plt.subplots_adjust(bottom=0.32, wspace=0.3)
plt.rcParams["boxplot.medianprops.color"] = "k"
plt.rcParams["boxplot.medianprops.linewidth"] = 3.0
outlier_porps = dict(markersize=3, markeredgecolor="black", marker='.')
results = run_models(use_cached=True)
for i, dataset in enumerate(DATASET_ORDER):
    # set dataset title
    axs[i].set_title(PLOT_CONFIG[dataset]["display_name"])
    mean_dice_overlaps = []
    labels = []
    label_colors = []
    label_bold_font = []
    colors = []
    for loss_function in LOSS_FUNTION_ORDER:
        # test model
        if loss_function in results[dataset].keys():
            mean_dice_overlaps.append(results[dataset][loss_function]["dice_overlap"])
            rank = results[dataset][loss_function]["rank"]
            if rank == 0:
                label_colors.append("black")
                label_bold_font.append(True)
                labels.append(LOSS_FUNTION_CONFIG[loss_function]["display_name_bold"])
            elif rank == 1:
                label_colors.append("black")
                label_bold_font.append(False)
                labels.append(LOSS_FUNTION_CONFIG[loss_function]["display_name"])
            else:
                label_colors.append('#404040')
                label_bold_font.append(False)
                labels.append(LOSS_FUNTION_CONFIG[loss_function]["display_name"])
            colors.append(LOSS_FUNTION_CONFIG[loss_function]["primary_color"])

    # plot boxes.
    if len(labels) > 0:
        bplot = axs[i].boxplot(
            np.array(mean_dice_overlaps).T,
            vert=True,  # vertical box alignment
            patch_artist=True,  # fill with color
            labels=labels,
            flierprops=outlier_porps
        )  # will be used to label x-ticks
        # color boxes
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)

        # rotate labels
        for tick, color in zip(axs[i].get_xticklabels(), label_colors):
            tick.set_rotation(60)
            tick.set_color(color)
            
    
# configure axis precision
for ax in axs:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


# add labels
fig.text(0.06, 0.5, "Test Mean Dice Overlap", ha="center", va="center", rotation="vertical")

os.makedirs("./out/plots/", exist_ok=True)
plt.savefig("./out/plots/test_score.pdf")
plt.savefig("./out/plots/test_score.png")

