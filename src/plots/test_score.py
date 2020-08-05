import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from src.registration_model import RegistrationModel
import torch
from .config import *
from .run_models import run_models

# set up sup-plots
fig = plt.figure(figsize=(12,4))
axs = fig.subplots(1, len(DATASET_ORDER)) 
plt.subplots_adjust(bottom=0.2)
plt.rcParams['boxplot.medianprops.color'] = 'k'
plt.rcParams['boxplot.medianprops.linewidth'] = 3.0
results = run_models(use_cached=True)
for i, dataset in enumerate(DATASET_ORDER):
    # set dataset title
    axs[i].set_title(PLOT_CONFIG[dataset]['display_name'])
    mean_dice_overlaps = []
    labels = []
    colors = []
    for loss_function in LOSS_FUNTION_ORDER:
        # test model
        if loss_function in results[dataset].keys():
            mean_dice_overlaps.append(results[dataset][loss_function]["dice_overlap"])
            labels.append(LOSS_FUNTION_CONFIG[loss_function]['display_name'])
            colors.append(LOSS_FUNTION_CONFIG[loss_function]['primary_color'])

    # plot boxes.
    if len(labels) > 0:
        bplot = axs[i].boxplot(np.array(mean_dice_overlaps).T,
                    vert=True,  # vertical box alignment
                    patch_artist=True,  # fill with color
                    labels=labels) # will be used to label x-ticks
        # color boxes
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        # rotate labels
        for tick in axs[i].get_xticklabels():
            tick.set_rotation(70)


# add labels
fig.text(0.06, 0.5, 'Mean Dice Overlap', ha='center', va='center', rotation='vertical')

os.makedirs('./out/plots/', exist_ok=True)
plt.savefig('./out/plots/test_score.pdf')
plt.savefig('./out/plots/test_score.png')