import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from src.registration_model import RegistrationModel
import torch
from .config import *


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
fig = plt.figure(figsize=(10,4))
axs = fig.subplots(1, len(DATASET_ORDER)) 
plt.subplots_adjust(bottom=0.2)
plt.rcParams['boxplot.medianprops.color'] = 'k'
plt.rcParams['boxplot.medianprops.linewidth'] = 3.0

for i, dataset in enumerate(DATASET_ORDER):
    # set dataset title
    axs[i].set_title(PLOT_CONFIG[dataset]['display_name'])
    mean_dice_overlaps = []
    labels = []
    colors = []
    for loss_function in tqdm(LOSS_FUNTION_ORDER):
        path = os.path.join('./weights/', dataset, 'registration', loss_function)
        if not os.path.isdir(path):
            continue
        # load model
        checkpoint_path = os.path.join(path, 'weights.ckpt')
        model = RegistrationModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path)
        # test model
        mean_dice_overlaps.append(test_model(model))
        #mean_dice_overlaps.append(list(np.random.normal(0.8, 0.2, 50)))
        labels.append(LOSS_FUNTION_CONFIG[loss_function]['display_name'])
        colors.append(LOSS_FUNTION_CONFIG[loss_function]['primary_color'])

    # plot
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

plt.savefig('./src/plots/test_score.pdf')
plt.savefig('./src/plots/test_score.png')