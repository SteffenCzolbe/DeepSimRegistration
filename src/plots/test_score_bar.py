import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from src.registration_model import RegistrationModel
import torch
from .config import *
from .run_models import run_models


# read logs
def test_model(model):
    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.eval()
        model = model.to(device)
        test_set = model.test_dataloader().dataset

        scores = []
        for i in range(len(test_set)):
            (I_0, S_0), (I_1, S_1) = test_set[i]
            batch = (I_0.unsqueeze(0).to(device), S_0.unsqueeze(0).to(device)), (I_1.unsqueeze(0).to(device), S_1.unsqueeze(0).to(device))
            score = model._step(batch, None)
            scores.append(score["dice_overlap"].item())
    return scores

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
        if loss_function in results[dataset].keys():
            mean_dice_overlaps.append(results[dataset][loss_function]["dice_overlap"].mean())
            labels.append(LOSS_FUNTION_CONFIG[loss_function]['display_name'])
            colors.append(LOSS_FUNTION_CONFIG[loss_function]['primary_color'])

    # plot bars.
    if len(labels) > 0:
        bplot = axs[i].bar(np.arange(len(mean_dice_overlaps)) + 0.5, mean_dice_overlaps,
                    tick_label=labels, color=colors)
        data_range = np.max(mean_dice_overlaps) - np.min(mean_dice_overlaps)
        ymin = np.min(mean_dice_overlaps) - 0.5 * data_range
        ymax = np.max(mean_dice_overlaps) + 0.1 * data_range
        axs[i].set_ylim([ymin,ymax])

        # rotate labels
        for tick in axs[i].get_xticklabels():
            tick.set_rotation(70)


# add labels
fig.text(0.06, 0.5, 'Mean Dice Overlap', ha='center', va='center', rotation='vertical')

os.makedirs('./out/plots/', exist_ok=True)
plt.savefig('./out/plots/test_score_bar.pdf')
plt.savefig('./out/plots/test_score_bar.png')