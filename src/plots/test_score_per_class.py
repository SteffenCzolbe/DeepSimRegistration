import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm

from .config import *
from .run_models import run_models
from src.registration_model import RegistrationModel

dataset = "brain-mri"

# set up sup-plots
fig = plt.figure(figsize=(10, 4))
ax = fig.subplots(1)
plt.subplots_adjust(bottom=0.2)
plt.rcParams["boxplot.medianprops.color"] = "k"
plt.rcParams["boxplot.medianprops.linewidth"] = 1.0
plt.rcParams["boxplot.flierprops.markeredgecolor"] = "k"
plt.rcParams["boxplot.flierprops.markeredgewidth"] = 0.25
plt.rcParams["boxplot.flierprops.markersize"] = 3
plt.rcParams["boxplot.flierprops.linewidth"] = 0.5
plt.rcParams["boxplot.boxprops.linewidth"] = 0.5
plt.rcParams["boxplot.whiskerprops.linewidth"] = 0.5
plt.rcParams["boxplot.capprops.linewidth"] = 0.5
plt.grid(linestyle="--", linewidth=0.5)
plt.tight_layout(pad=5, h_pad=None, w_pad=None, rect=(0, 0.2, 1, 1.1))
plt.ylim(0, 1)
results = run_models(use_cached=True)

dice_overlap_of_classes = []
median_dice_overlap_of_classes = []
labels = []
bold_labels = []
label_colors = []
colors = []
classes = list(range(results[dataset]["l2"]
                     ["dice_overlap_per_class"].shape[1]))
class_to_name_dict = json.loads(
    open("./src/plots/brain_mri_labels.json").read())

# order by decreasing scores of first loss function
mean_dice_overlaps = results[dataset][LOSS_FUNTION_ORDER[0]][
    "dice_overlap_per_class"
].mean(axis=0)
classes = np.argsort(-mean_dice_overlaps).tolist()
# remove unwanted classes: background (0), 5th ventricle (21), hyperintensity (23), vessel (19)
for rm_class in [0, 21, 23, 19]:
    classes.remove(rm_class)

def make_bold(means):
    # decide wich labels to make bold
    ours = [3, 4]  # 0-indexed positions of our metrics
    order = np.flip(np.argsort(means))
    order_ours = [i in ours for i in order]
    return order_ours[0] and order_ours[1]


# aggregate data
for c in classes:
    for loss_function in LOSS_FUNTION_ORDER:
        if loss_function in results[dataset].keys():
            dice_overlap_of_classes.append(
                results[dataset][loss_function]["dice_overlap_per_class"][:, c]
            )
            median_dice_overlap_of_classes.append(
                np.mean(results[dataset][loss_function]
                        ["dice_overlap_per_class"][:, c])
            )
            colors.append(LOSS_FUNTION_CONFIG[loss_function]["primary_color"])
    # set tick labels
    col_count = len(results[dataset].keys())
    for _ in range((col_count // 2) + 1):
        labels.append("")
    if make_bold(median_dice_overlap_of_classes[-5:]):
        #labels.append(r"\textbf{" + class_to_name_dict[str(c)] + "}")
        labels.append("$\\bf{" + str(class_to_name_dict[str(c)]) + "}$")
         #r"$\bf{" + str(number) + "}$"
        bold_labels.append( class_to_name_dict[str(c)])
    else:
        labels.append(class_to_name_dict[str(c)])

    for _ in range(col_count - (col_count // 2) - 1):
        labels.append("")
    # pad emplty col
    colors.append("black")
    dice_overlap_of_classes.append([])

# plot boxes.
if len(labels) > 0:
    bplot = ax.boxplot(
        np.array(dice_overlap_of_classes).T,
        vert=True,  # vertical box alignment
        patch_artist=True,  # fill with color
        labels=labels,
    )  # will be used to label x-ticks
    handles = bplot["boxes"]
    # color boxes
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
    
    # rotate labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=70,
             ha="right", rotation_mode="anchor")

# add legend
labels = [LOSS_FUNTION_CONFIG[l]["display_name"] for l in LOSS_FUNTION_ORDER]

#prop={"size": 7, "weight":'bold'}
prop={"size": 7}
ax.legend(handles[-(1+len(LOSS_FUNTION_ORDER)):-1],
          labels, loc="lower left", prop=prop)

# add labels
fig.text(0.06, 0.675, "Dice Overlap", ha="center",
         va="center", rotation="vertical")

os.makedirs("./out/plots/pdf/", exist_ok=True)
os.makedirs("./out/plots/png/", exist_ok=True)
plt.savefig(f"./out/plots/pdf/test_score_per_class_{dataset}.pdf")
plt.savefig(f"./out/plots/png/test_score_per_class_{dataset}.png")