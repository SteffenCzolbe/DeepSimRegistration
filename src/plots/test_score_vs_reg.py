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
        regularization = []
        for i in range(len(test_set)):
            (I_0, S_0), (I_1, S_1) = test_set[i]
            batch = (I_0.unsqueeze(0).to(device), S_0.unsqueeze(0).to(device)), (I_1.unsqueeze(0).to(device), S_1.unsqueeze(0).to(device))
            score = model._step(batch, None)
            scores.append(score["dice_overlap"].item())
            regularization.append(score["regularization"].item())
    return np.mean(scores), np.mean(regularization)

# set up sup-plots
fig = plt.figure(figsize=(12,4))
axs = fig.subplots(1, len(DATASET_ORDER)) 
plt.subplots_adjust(bottom=0.2)
plt.rcParams['boxplot.medianprops.color'] = 'k'
plt.rcParams['boxplot.medianprops.linewidth'] = 3.0

for i, dataset in enumerate(DATASET_ORDER):
    # set dataset title
    axs[i].set_title(PLOT_CONFIG[dataset]['display_name'])
    dice_overlaps = []
    regularizations = []
    labels = []
    colors = []
    for loss_function in tqdm(LOSS_FUNTION_ORDER, desc=f'testing loss-finctions on {dataset}'):
        path = os.path.join('./weights/', dataset, 'registration', loss_function)
        if not os.path.isdir(path):
            continue
        # load model
        checkpoint_path = os.path.join(path, 'weights.ckpt')
        model = RegistrationModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path)
        # test model
        dice_overlap, regularization = test_model(model)
        dice_overlaps.append(np.mean(dice_overlap))
        regularizations.append(np.mean(regularization))
        labels.append(LOSS_FUNTION_CONFIG[loss_function]['display_name'])
        colors.append(LOSS_FUNTION_CONFIG[loss_function]['primary_color'])

    # plot boxes.
    if len(labels) > 0:
        for j in range(len(labels)):
            axs[i].scatter(regularizations[j], dice_overlaps[j], color=colors[j])
            axs[i].annotate(labels[j], (regularizations[j], dice_overlaps[j]))


# add labels
fig.text(0.5, 0.02, 'Regularization', ha='center', va='center')
fig.text(0.06, 0.5, 'Mean Dice Overlap', ha='center', va='center', rotation='vertical')

os.makedirs('./out/plots/', exist_ok=True)
plt.savefig('./out/plots/test_score_vs_reg.pdf')
plt.savefig('./out/plots/test_score_vs_reg.png')