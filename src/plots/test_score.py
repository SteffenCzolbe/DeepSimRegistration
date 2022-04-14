import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
from tqdm import tqdm
from src.registration_model import RegistrationModel
import torch
from .config import *
from .run_models import run_models
from matplotlib.ticker import FormatStrFormatter


def get_score_and_label(dataset, loss_function):
    if loss_function is None:
        return get_empty_score_and_label()
    if loss_function in LOSS_FUNTION_ORDER:
        return get_score_and_label_for_dl_method(dataset, loss_function)
    else:
        return get_score_and_label_for_syn(dataset, loss_function)


def get_empty_score_and_label():
    bar_color = "black"
    label = ""
    label_color = "black"
    scores = []
    return scores, bar_color, label, label_color


def get_score_and_label_for_syn(dataset, loss_function):
    dataset_path = os.path.join('./out', dataset, 'syn')
    fnames = os.listdir(dataset_path)
    scores = {}

    # read scores
    for fname in fnames:
        with open(os.path.join(dataset_path, fname), 'r') as file:
            data = yaml.safe_load(file)
            feature_extractor = data['hparams']['feature_extractor']
            scores[feature_extractor] = data['scores']

    # filter out images where one model is missing a score (helps with partial evaluation during development)
    feature_extractors = scores.keys()
    idxs = [set(scores[fe].keys()) for fe in feature_extractors]
    common_idxs = set.intersection(* idxs)
    for feature_extractor in feature_extractors:
        scores[feature_extractor] = np.array([scores[feature_extractor][idx]
                                              for idx in common_idxs])
    bar_color = LOSS_FUNTION_CONFIG[loss_function]["primary_color"]
    is_our_method = LOSS_FUNTION_CONFIG[loss_function]["our_method"]
    label = LOSS_FUNTION_CONFIG[loss_function]["display_name_bold" if is_our_method else "display_name"]
    label_color = "black"
    return scores[LOSS_FUNTION_CONFIG[loss_function]["feature_extractor"]], bar_color, label, label_color


def get_score_and_label_for_dl_method(dataset, loss_function):
    results = run_models(use_cached=True)
    #print(results)
    #print(type(results))
    if loss_function not in results[dataset].keys():
        return False
    scores = results[dataset][loss_function]["dice_overlap"]
    bar_color = LOSS_FUNTION_CONFIG[loss_function]["primary_color"]
    label_color = "black"
    is_our_method = LOSS_FUNTION_CONFIG[loss_function]["our_method"]
    label = LOSS_FUNTION_CONFIG[loss_function]["display_name_bold" if is_our_method else "display_name"]
    return scores, bar_color, label, label_color


# set up sup-plots
fig = plt.figure(figsize=(10, 3.5))
axs = fig.subplots(1, len(DATASET_ORDER))
plt.subplots_adjust(bottom=0.4, wspace=0.2)
plt.rcParams["boxplot.medianprops.color"] = "k"
plt.rcParams["boxplot.medianprops.linewidth"] = 3.0
outlier_porps = dict(markersize=3, markeredgecolor="black", marker='.')
for i, dataset in enumerate(DATASET_ORDER):
    # set dataset title
    axs[i].set_title(PLOT_CONFIG[dataset]["display_name"])
    mean_dice_overlaps = []
    labels = []
    label_colors = []
    label_bold_font = []
    colors = []
    for loss_function in ALL_METHODS:
        # test model
        r = get_score_and_label(dataset, loss_function)
        if not r:
            continue
        scores, bar_color, label, label_color = r
        mean_dice_overlaps.append(scores)
        colors.append(bar_color)
        labels.append(label)
        label_colors.append(label_color)

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

        # rotate and color labels
        for tick, color in zip(axs[i].get_xticklabels(), label_colors):
            tick.set_rotation(80)
            tick.set_color(color)

        # hide empty ticks
        for j, label in enumerate(labels):
            if label is None or label == "":
                axs[i].xaxis.majorTicks[j].set_visible(False)


# configure axis precision
for ax in axs:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


# add labels
fig.text(0.075, 0.63, "Test Mean Dice Overlap",
         ha="center", va="center", rotation="vertical")

os.makedirs("./out/plots/", exist_ok=True)
plt.savefig("./out/plots/test_score.pdf", bbox_inches='tight')
plt.savefig("./out/plots/test_score.png", bbox_inches='tight')
