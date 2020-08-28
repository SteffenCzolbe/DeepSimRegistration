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

def plot_brainmri(fig, row, col, I, S, inv_flow=None, title=None):
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

    # following operations are on a 2d slice
    torchreg.settings.set_ndims(2) 

    fig.plot_img(row, col, I[0], vmin=0, vmax=1, title=title)
    fig.plot_overlay_class_mask(row, col, S[0], num_classes=model.dataset_config('classes'), 
        colors=model.dataset_config('class_colors'), alpha=0.3)
    classes_to_annotate = list(map(lambda t: t[0], filter(lambda t: t[1] is not None, model.dataset_config('class_colors').items())))
    for c in classes_to_annotate:
        fig.plot_contour(row, col, S[0], contour_class=c, width=1, rgba=model.dataset_config('class_colors')[c])
    if inv_flow is not None:
        fig.plot_transform_grid(row, col, inv_flow[0], interval=7, linewidth=0.1, color='#FFFFFFFF' , overlay=True)

    torchreg.settings.set_ndims(3) # back to 3d


DATASET_ORDER = ['brain-mri']

fig = viz.Fig(1, 6, None, figsize=(10, 6))
for i, dataset in enumerate(DATASET_ORDER):

    j = 2
    for loss_function in LOSS_FUNTION_ORDER:
        path = os.path.join('./weights/', dataset, 'registration', loss_function)
        if not os.path.isdir(path):
            continue
        # load model
        checkpoint_path = os.path.join(path, 'weights.ckpt')
        model = RegistrationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

        # run model
        I_0, S_0, I_m, S_m, I_1, S_1, inv_flow = get_img(model, 0)

        # plot aligned image
        plot_brainmri(fig, i, j, I_m, S_m, inv_flow=inv_flow, title=LOSS_FUNTION_CONFIG[loss_function]['display_name'])
        j += 1

    # plot moved and fixed image
    plot_brainmri(fig, i, 0, I_0, S_0, title='$I_0$')
    plot_brainmri(fig, i, 1, I_1, S_1, title='$I_1$')

os.makedirs('./out/plots', exist_ok=True)
fig.save('./out/plots/brain_sample.pdf', close=False)
fig.save('./out/plots/brain_sample.png')


