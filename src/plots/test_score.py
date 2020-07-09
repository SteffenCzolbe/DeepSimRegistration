import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from src.registration_model import RegistrationModel
import torch


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

# read datasets
datasets = os.listdir('./weights/')

# set up sup-plots
fig = plt.figure()
axs = fig.subplots(1, len(datasets)) 
colors = plt.get_cmap('tab20').colors[1::2]
plt.rcParams['boxplot.medianprops.color'] = 'k'
plt.rcParams['boxplot.medianprops.linewidth'] = 3.0

for i, dataset in enumerate(datasets):
    loss_functions = sorted(os.listdir(os.path.join('./weights/', dataset, 'registration')))
    mean_dice_overlaps = []
    for loss_function in tqdm(loss_functions):
        # load model
        checkpoint_path = os.path.join('./weights/', dataset, 'registration', loss_function, 'weights.ckpt')
        model = RegistrationModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path)
        mean_dice_overlaps.append(test_model(model))

    # plot
    # plot boxes.
    bplot = axs[i].boxplot(np.array(mean_dice_overlaps).T,
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                labels=loss_functions) # will be used to label x-ticks
    # color boxes
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    # set dataset title
    axs[i].set_title(dataset)


# add labels
fig.text(0.06, 0.5, 'Mean Dice Overlap', ha='center', va='center', rotation='vertical')

plt.savefig('./src/plots/test_score.pdf')
plt.savefig('./src/plots/test_score.png')