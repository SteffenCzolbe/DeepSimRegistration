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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.eval()
        model = model.to(device)
        test_set = model.test_dataloader().dataset
        (I_0, S_0), (I_1, S_1) = test_set[test_set_index]
        (I_0, S_0), (I_1, S_1) = (I_0.unsqueeze(0).to(device), S_0.unsqueeze(0).to(device)), (I_1.unsqueeze(0).to(device), S_1.unsqueeze(0).to(device))
        flow = model.forward(I_0, I_1)
        I_m = transformer(I_0, flow)
        S_m = transformer(S_0.float(), flow, mode='nearest').round().long()
        inv_flow = integrate(-flow)
    return I_0, S_0, I_m, S_m, I_1, S_1, inv_flow
    

def crop(x_low, x_high, y_low, y_high, I):
    return I[:,:, x_low:x_high, y_low:y_high] 

def plot_platelet(fig, row, col, I, S, inv_flow=None):
    I = crop(270, 470, 100, 300, I)
    S = crop(270, 470, 100, 300, S)
    if inv_flow is not None:
        inv_flow = crop(270, 470, 100, 300, inv_flow)

    fig.plot_img(row, col, I[0], vmin=0, vmax=1)
    fig.plot_overlay_class_mask(row, col, S[0], num_classes=model.dataset_config('classes'), 
        colors=model.dataset_config('class_colors'), alpha=0.2)
    fig.plot_contour(row, col, S[0], contour_class=1, width=2, rgba=model.dataset_config('class_colors')[1])
    fig.plot_contour(row, col, S[0], contour_class=2, width=2, rgba=model.dataset_config('class_colors')[2])
    if inv_flow is not None:
        fig.plot_transform_grid(row, col, inv_flow[0], interval=14, linewidth=0.2, color='#FFFFFFFF' , overlay=True)

def plot_phc(fig, row, col, I, S, inv_flow=None):
    I = crop(200, 300, 0, 100, I)
    S = crop(200, 300, 0, 100, S)
    if inv_flow is not None:
        inv_flow = crop(200, 300, 0, 100, inv_flow)

    fig.plot_img(row, col, I[0], vmin=0, vmax=1)
    fig.plot_overlay_class_mask(row, col, S[0], num_classes=model.dataset_config('classes'), 
        colors=model.dataset_config('class_colors'), alpha=0.2)
    fig.plot_contour(row, col, S[0], contour_class=1, width=1, rgba=model.dataset_config('class_colors')[1])
    if inv_flow is not None:
        fig.plot_transform_grid(row, col, inv_flow[0], interval=8, linewidth=0.2, color='#FFFFFFFF' , overlay=True)

def plot_brainmri(fig, row, col, I, S, inv_flow=None):
    # get slice
    I = I[:,:, :, :, 100]
    S = S[:,:, :, :, 100]
    if inv_flow is not None:
        inv_flow = inv_flow[:,:2, :, :, 100]


    # rotate by 90 degree right
    I = I.permute(0,1,3,2)
    S = S.permute(0,1,3,2)
    if inv_flow is not None:
        inv_flow = inv_flow.permute(0,1,3,2)
        inv_flow = torch.stack([inv_flow[:, 1], inv_flow[:, 0]], dim=1)

    I = crop(0, 160, 0, 160, I)
    S = crop(0, 160, 0, 160, S)
    if inv_flow is not None:
        inv_flow = crop(0, 160, 0, 160, inv_flow)

    # following operations are on a 2d slice
    torchreg.settings.set_ndims(2) 

    fig.plot_img(row, col, I[0], vmin=0, vmax=1)
    fig.plot_overlay_class_mask(row, col, S[0], num_classes=model.dataset_config('classes'), 
        colors=model.dataset_config('class_colors'), alpha=0.3)
    classes_to_annotate = list(map(lambda t: t[0], filter(lambda t: t[1] is not None, model.dataset_config('class_colors').items())))
    for c in classes_to_annotate:
        fig.plot_contour(row, col, S[0], contour_class=c, width=1, rgba=model.dataset_config('class_colors')[c])
    if inv_flow is not None:
        fig.plot_transform_grid(row, col, inv_flow[0], interval=7, linewidth=0.1, color='#FFFFFFFF' , overlay=True)

    torchreg.settings.set_ndims(3) # back to 3d


DATASET_ORDER = ['brain-mri', 'platelet-em', 'phc-u373']

fig = viz.Fig(5, 8, None, figsize=(8, 5))
# adjust subplot spacing
fig.fig.subplots_adjust(hspace=0.05, wspace=0.05)

for i, dataset in enumerate(DATASET_ORDER):
    # set plotting function
    if dataset == 'platelet-em':
        plotfun = plot_platelet
        sample_idx = 5
    elif dataset == 'brain-mri':
        plotfun = plot_brainmri
        sample_idx = 0
    elif dataset == 'phc-u373':
        plotfun = plot_phc
        sample_idx = 9

    for j, loss_function in enumerate(LOSS_FUNTION_ORDER):
        path = os.path.join('./weights/', dataset, 'registration', loss_function)
        if not os.path.isdir(path):
            continue
        # load model
        checkpoint_path = os.path.join(path, 'weights.ckpt')
        model = RegistrationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

        # run model
        I_0, S_0, I_m, S_m, I_1, S_1, inv_flow = get_img(model, sample_idx)

        # plot aligned image
        plotfun(fig, i, j+2, I_m, S_m, inv_flow=inv_flow)

    # plot moved and fixed image
    plotfun(fig, i, 0, I_0, S_0)
    plotfun(fig, i, 1, I_1, S_1)

# label loss function
for i, lossfun in enumerate(LOSS_FUNTION_ORDER):
    fig.axs[0, i+2].set_title(LOSS_FUNTION_CONFIG[lossfun]['display_name'])
fig.axs[0, 0].set_title(r'$\mathbf{I}$')
fig.axs[0, 1].set_title(r'$\mathbf{J}$')

os.makedirs('./out/plots', exist_ok=True)
fig.save('./out/plots/img_sample.pdf', close=False)
fig.save('./out/plots/img_sample.png')


