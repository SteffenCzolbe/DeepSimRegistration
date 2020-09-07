import pickle
import os
import numpy as np
from tqdm import tqdm
from src.registration_model import RegistrationModel
import torch
import torchreg
import torchreg.viz as viz
from .config import *


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


def plot_platelet(fig, row, col, I, S, inv_flow=None, title=None):
    crop_area = (420, 600, 500, 680)
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
            interval=10,
            linewidth=0.5,
            color="#000000FF",
            overlay=True,
        )


dataset = "platelet-em"
sample_idx = 10
LOSS_FUNTION_ORDER = ["ncc2", "deepsim"]

fig = viz.Fig(1, 2, None, figsize=(7, 4))
# adjust subplot spacing
fig.fig.subplots_adjust(hspace=0.05, wspace=0.05)

# set plotting function
plotfun = plot_platelet

for j, loss_function in enumerate(LOSS_FUNTION_ORDER):
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
        0,
        j,
        I_m,
        S_m,
        inv_flow=inv_flow,
        title=LOSS_FUNTION_CONFIG[loss_function]["display_name"],
    )

# plot moved and fixed image
# plotfun(fig, sample_idx, 0, I_0, S_0)
# plotfun(fig, sample_idx, 1, I_1, S_1)


os.makedirs("./out/plots", exist_ok=True)
fig.save("./out/plots/platelet_sample.pdf", close=False)
fig.save("./out/plots/platelet_sample.png")

