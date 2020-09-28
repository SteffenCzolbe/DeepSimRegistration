import pickle
import os
import numpy as np
from tqdm import tqdm
from src.registration_model import RegistrationModel
import torch
import torchreg
import torchreg.viz as viz
from .config import *
import matplotlib.patches as patches


def get_img(model, test_set_index):
    transformer = torchreg.nn.SpatialTransformer()
    integrate = torchreg.nn.FlowIntegration(6)

    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        model = model.to(device)
        test_set = model.test_dataloader().dataset
        (I_0, S_0), (I_1, S_1) = test_set[test_set_index]
        (I_0, S_0), (I_1, S_1) = (
            (I_0.unsqueeze(0).to(device), S_0.unsqueeze(0).to(device)),
            (I_1.unsqueeze(0).to(device), S_1.unsqueeze(0).to(device)),
        )
        flow = model.forward(I_0, I_1)
        I_m = transformer(I_0, flow)
        S_m = transformer(S_0.float(), flow, mode="nearest").round().long()
        inv_flow = integrate(-flow)
    return I_0, S_0, I_m, S_m, I_1, S_1, inv_flow


def crop(x_low, x_high, y_low, y_high, I):
    return I[:, :, x_low:x_high, y_low:y_high]


def plot_platelet(fig, row, col, model, I, S, inv_flow=None, title=None, highlight_color=None):
    crop_area = (270, 470, 100, 300)
    highlight_area = (350, 410, 102, 160)
    I = crop(*crop_area, I)
    S = crop(*crop_area, S)
    if inv_flow is not None:
        inv_flow = crop(*crop_area, inv_flow)

    fig.plot_img(row, col, I[0], vmin=0, vmax=1, title=title)
    fig.plot_overlay_class_mask(
        row,
        col,
        S[0],
        num_classes=model.dataset_config("classes"),
        colors=model.dataset_config("class_colors"),
        alpha=0.2,
    )
    fig.plot_contour(
        row,
        col,
        S[0],
        contour_class=1,
        width=2,
        rgba=model.dataset_config("class_colors")[1],
    )
    fig.plot_contour(
        row,
        col,
        S[0],
        contour_class=2,
        width=2,
        rgba=model.dataset_config("class_colors")[2],
    )
    if inv_flow is not None:
        fig.plot_transform_grid(
            row,
            col,
            inv_flow[0],
            interval=14,
            linewidth=0.2,
            color="#FFFFFFFF",
            overlay=True,
        )
    if highlight_color is not None:
        # add highlight-frame
        h = highlight_area[1] - highlight_area[0]
        w = highlight_area[3] - highlight_area[2]
        p = (highlight_area[2] - crop_area[2], highlight_area[0] - crop_area[0])
        rect = patches.Rectangle(p,h,w,linewidth=2,edgecolor=highlight_color,facecolor='none')
        fig.axs[row, col].add_patch(rect)
    
        
def plot_platelet_detail(fig, row, col, model, I, S, inv_flow=None, title=None, highlight_color=None):
    crop_area = (350, 410, 100, 160)
    highlight_area = (350, 409, 100, 159)
    I = crop(*crop_area, I)
    S = crop(*crop_area, S)
    if inv_flow is not None:
        inv_flow = crop(*crop_area, inv_flow)

    fig.plot_img(row, col, I[0], vmin=0, vmax=1, title=title)
    fig.plot_contour(
        row,
        col,
        S[0],
        contour_class=1,
        width=2,
        rgba=model.dataset_config("class_colors")[1],
    )
    if inv_flow is not None:
        fig.plot_transform_grid(
            row,
            col,
            inv_flow[0],
            interval=7,
            linewidth=0.5,
            color="#000000FF",
            overlay=True,
        )
    if highlight_color is not None:
        # add highlight-frame
        h = highlight_area[1] - highlight_area[0]
        w = highlight_area[3] - highlight_area[2]
        p = (highlight_area[2] - crop_area[2], highlight_area[0] - crop_area[0])
        rect = patches.Rectangle(p,h,w,linewidth=4,edgecolor=highlight_color,facecolor='none')
        fig.axs[row, col].add_patch(rect)


def plot_phc(fig, row, col, model, I, S, inv_flow=None):
    I = crop(200, 300, 0, 100, I)
    S = crop(200, 300, 0, 100, S)
    if inv_flow is not None:
        inv_flow = crop(200, 300, 0, 100, inv_flow)

    fig.plot_img(row, col, I[0], vmin=0, vmax=1)
    fig.plot_overlay_class_mask(
        row,
        col,
        S[0],
        num_classes=model.dataset_config("classes"),
        colors=model.dataset_config("class_colors"),
        alpha=0.2,
    )
    fig.plot_contour(
        row,
        col,
        S[0],
        contour_class=1,
        width=1,
        rgba=model.dataset_config("class_colors")[1],
    )
    if inv_flow is not None:
        fig.plot_transform_grid(
            row,
            col,
            inv_flow[0],
            interval=8,
            linewidth=0.2,
            color="#FFFFFFFF",
            overlay=True,
        )


def plot_brainmri(fig, row, col, model, I, S, inv_flow=None):
    # get slice
    I = I[:, :, :, :, 100]
    S = S[:, :, :, :, 100]
    if inv_flow is not None:
        inv_flow = inv_flow[:, :2, :, :, 100]

    # rotate by 90 degree right
    I = I.permute(0, 1, 3, 2)
    S = S.permute(0, 1, 3, 2)
    if inv_flow is not None:
        inv_flow = inv_flow.permute(0, 1, 3, 2)
        inv_flow = torch.stack([inv_flow[:, 1], inv_flow[:, 0]], dim=1)

    I = crop(0, 160, 0, 160, I)
    S = crop(0, 160, 0, 160, S)
    if inv_flow is not None:
        inv_flow = crop(0, 160, 0, 160, inv_flow)

    # following operations are on a 2d slice
    torchreg.settings.set_ndims(2)

    fig.plot_img(row, col, I[0], vmin=0, vmax=1)
    fig.plot_overlay_class_mask(
        row,
        col,
        S[0],
        num_classes=model.dataset_config("classes"),
        colors=model.dataset_config("class_colors"),
        alpha=0.3,
    )
    classes_to_annotate = list(
        map(
            lambda t: t[0],
            filter(
                lambda t: t[1] is not None, model.dataset_config("class_colors").items()
            ),
        )
    )
    for c in classes_to_annotate:
        fig.plot_contour(
            row,
            col,
            S[0],
            contour_class=c,
            width=1,
            rgba=model.dataset_config("class_colors")[c],
        )
    if inv_flow is not None:
        fig.plot_transform_grid(
            row,
            col,
            inv_flow[0],
            interval=10,
            linewidth=0.15,
            color="#FFFFFFFF",
            overlay=True,
        )

    torchreg.settings.set_ndims(3)  # back to 3d


def make_overview():
    fig = viz.Fig(5, 8, None, figsize=(8, 5))
    # adjust subplot spacing
    fig.fig.subplots_adjust(hspace=0.05, wspace=0.05)
    highlight_colors = [None, 'r', None, None, '#31e731']

    for i, dataset in enumerate(DATASET_ORDER):
        # set plotting function
        if dataset == "platelet-em":
            plotfun = plot_platelet
            sample_idx = 5
        elif dataset == "brain-mri":
            plotfun = plot_brainmri
            sample_idx = 0
        elif dataset == "phc-u373":
            plotfun = plot_phc
            sample_idx = 9

        for j, (loss_function, highlight_color) in enumerate(zip(LOSS_FUNTION_ORDER, highlight_colors)):
            path = os.path.join("./weights/", dataset, "registration", loss_function)
            if not os.path.isdir(path):
                continue
            # load model
            checkpoint_path = os.path.join(path, "weights.ckpt")
            model = RegistrationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

            # run model
            I_0, S_0, I_m, S_m, I_1, S_1, inv_flow = get_img(model, sample_idx)

            # plot aligned image
            kwargs = {'highlight_color': highlight_color} if dataset == "platelet-em" else {}
            plotfun(fig, i, j + 2, model, I_m, S_m, inv_flow=inv_flow, **kwargs)

        # plot moved and fixed image
        plotfun(fig, i, 0, model, I_0, S_0)
        plotfun(fig, i, 1, model, I_1, S_1)

    # label loss function
    for i, lossfun in enumerate(LOSS_FUNTION_ORDER):
        fig.axs[0, i + 2].set_title(LOSS_FUNTION_CONFIG[lossfun]["display_name"])
    fig.axs[0, 0].set_title("Moving")
    fig.axs[0, 1].set_title("Fixed")

    os.makedirs("./out/plots", exist_ok=True)
    fig.save("./out/plots/short_paper_img_sample.pdf", close=False)
    fig.save("./out/plots/short_paper_img_sample.png")



def make_detail():
    # detail view
    fig = viz.Fig(2, 1, None, figsize=(1.5, 3))
    # adjust subplot spacing
    fig.fig.subplots_adjust(hspace=0.3, wspace=0.05)

    # set plotting function
    plotfun = plot_platelet_detail
    dataset = "platelet-em"
    sample_idx = 5
    LOSS_FUNTION_ORDER = ["ncc2", "deepsim"]
    highlight_colors = ['r', '#31e731']

    for j, (loss_function, highlight_color) in enumerate(zip(LOSS_FUNTION_ORDER, highlight_colors)):
        path = os.path.join("./weights/", dataset, "registration", loss_function)
        if not os.path.isdir(path):
            continue
        # load model
        checkpoint_path = os.path.join(path, "weights.ckpt")
        model = RegistrationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

        # run model
        I_0, S_0, I_m, S_m, I_1, S_1, inv_flow = get_img(model, sample_idx)
        

        # plot aligned image
        plotfun(
            fig,
            j,
            0,
            model, 
            I_m,
            S_m,
            inv_flow=inv_flow,
            title=LOSS_FUNTION_CONFIG[loss_function]["display_name"],
            highlight_color=highlight_color
        )


    os.makedirs("./out/plots", exist_ok=True)
    fig.save("./out/plots/short_paper_img_sample_detail.pdf", close=False)
    fig.save("./out/plots/short_paper_img_sample_detail.png")


make_overview()
make_detail()

