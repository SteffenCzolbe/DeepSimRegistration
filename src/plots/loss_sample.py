import matplotlib.patches as patches
import numpy as np
import os
import pickle
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchreg
import torchreg.viz as viz
from scipy.stats.stats import pearsonr

from .config import *
from src.test_registration_voxelmorph import RegistrationModel
from src.segmentation_model import SegmentationModel
from src.autoencoder_model import AutoEncoderModel
from src.loss_metrics import NCC, DeepSim, VGGFeatureExtractor, NMI, MIND_loss
import os

LOSS_FUNTION_ORDER.remove('mind')
LOSS_FUNTION_ORDER.remove("nmi")
LOSS_FUNTION_ORDER.remove("ncc2+supervised")
  
def get_img(model, test_set_index):
    transformer = torchreg.nn.SpatialTransformer()
    integrate = torchreg.nn.FlowIntegration(6)

    with torch.no_grad():
        device = 'cpu'
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

def get_loss(dataset, loss_function, I_0, S_0, I_m, S_m, I_1, S_1):
    if loss_function =='deepsim':
        feature_extractor = SegmentationModel.load_from_checkpoint(f'./weights/{dataset}/segmentation/weights.ckpt')
        loss = DeepSim(feature_extractor, reduction='none')
        loss_per_pixel = loss(I_m, I_1)
    elif loss_function =='deepsim-ae':
        feature_extractor = AutoEncoderModel.load_from_checkpoint(f'./weights/{dataset}/autoencoder/weights.ckpt')
        loss = DeepSim(feature_extractor, reduction='none')
        loss_per_pixel = loss(I_1, I_m)
    elif loss_function == "ncc2":
        loss = NCC(window=9, squared=True, reduction='none')
        loss_per_pixel = loss(I_m, I_1)
        #loss_per_pixel = loss_per_pixel.permute(0,1,3,2)
    elif loss_function =='l2':
        loss = torch.nn.MSELoss(reduction='none')
        loss_per_pixel = loss(I_m, I_1)
    return loss_per_pixel

def my_plot(fig, row, col, model, I, title, cmap='gray', vmin=0, vmax=1, interpolation='none'):
    fig.plot_img(row, col, I[0], cmap=cmap, title=title, 
                 vmin=vmin, vmax=vmax,interpolation=interpolation)
                 
def make_loss_A():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 6})

    figA = viz.Fig(1, len(LOSS_FUNTION_ORDER) + 2, figsize=(7, 1))
    figA.fig.subplots_adjust(hspace=0.2, wspace=0.05)

    # set plotting function
    plotfun = my_plot
    dataset = "platelet-em"
    sample_idx = 5
    cmap = 'summer'

    for j, loss_function in enumerate(LOSS_FUNTION_ORDER):
        path = os.path.join("./weights/", dataset, "registration", loss_function)
        if not os.path.isdir(path):
            continue
        # load model
        checkpoint_path = os.path.join(path, "weights.ckpt")
        model = RegistrationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
        seg = '{seg}'
        ae = '{ae}'

        # run model
        I_0, S_0, I_m, S_m, I_1, S_1, inv_flow = get_img(model, sample_idx)
        crop_area = (270, 470, 100, 300)
        highlight_area = (350, 410, 102, 160)
        I_0 = crop(*crop_area, I_0)
        I_1 = crop(*crop_area, I_1)
        I_m = crop(*crop_area, I_m)

        plotfun(figA, 0, 0, model, I_0 ,title='Moving', vmin=0, vmax=1)
        plotfun(figA, 0, 1, model, I_1 ,title='Fixed', vmin=0, vmax=1)
                    
        lossy = get_loss(dataset, loss_function, I_0, S_0, I_m, S_m, I_1, S_1)

        if 'deepsim' in loss_function:
            seg_outputs = []
            ae_outputs = []

            for depth, l in enumerate(lossy):
                # pre-processing
                lo = -l + 1
                lo = torch.log(1 + lo)
                lo -= lo.min()
                lo /= lo.max()

                vmin, vmax = lo.min().item(), lo.max().item()
                if loss_function == 'deepsim':
                    seg_out = F.interpolate(lo.unsqueeze(0), size=(200, 200), mode='bilinear', align_corners=False).squeeze(0)
                    seg_outputs.append(seg_out)
                else:
                    ae_out = F.interpolate(lo.unsqueeze(0), size=(200, 200), mode='bilinear', align_corners=False).squeeze(0)
                    ae_outputs.append(ae_out)

            if loss_function == 'deepsim':
                seg_lvl = torch.stack(seg_outputs)
                mean_seg_lvl = torch.mean(seg_lvl, dim = 0)
            else:
                ae_lvl = torch.stack(ae_outputs)
                mean_ae_lvl = torch.mean(ae_lvl, dim = 0)

        else:
            if loss_function == "l2":
                loss = torch.log(1 + lossy)
                loss -= loss.min()
                loss /= loss.max()
                vmin = loss.min().item()
                vmax = loss.max().item() / 2
            else:
                loss = -lossy + 1
                loss = torch.log(1 + loss)
                loss -= loss.min()
                loss /= loss.max()
                vmin, vmax = loss.min().item(), loss.max().item()

            plotfun(figA, 0, j+2, model, loss,
                    title=LOSS_FUNTION_CONFIG[loss_function]["display_name"],
                    vmin=vmin, vmax=vmax,cmap=cmap)

    # print(mean_seg_lvl.size(), mean_ae_lvl.size())
    # print(mean_seg_lvl.min(), mean_seg_lvl.max(), mean_seg_lvl.mean())

    figA.plot_img(0, 4, mean_ae_lvl, cmap=cmap, title=f"$DeepSim_{ae}$", 
                  vmin=mean_ae_lvl.min(), vmax=mean_ae_lvl.max(),interpolation='bilinear')
    figA.plot_img(0, 5, mean_seg_lvl, cmap=cmap, title=f"$DeepSim_{seg}$", 
                  vmin=mean_seg_lvl.min(), vmax=mean_seg_lvl.max(),interpolation='bilinear')

    # calculate correlation between deepsim-seg and deepsim-ae
    seg_np = mean_seg_lvl.numpy().reshape(-1)
    ae_np = mean_ae_lvl.numpy().reshape(-1)
    corr = np.corrcoef(seg_np, ae_np)
    print(f'Correlation between deepsim-seg and deepsim-ae: {corr[0,1]}')

    os.makedirs("./out/plots/pdf/", exist_ok=True)
    os.makedirs("./out/plots/png/", exist_ok=True)

    figA.save("./out/plots/pdf/loss_sampleA.pdf", close=False)
    figA.save("./out/plots/png/loss_sampleA.png")

def make_loss_B():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 6})

    figB = viz.Fig(1, len(LOSS_FUNTION_ORDER) + 2, figsize=(7, 1))
    figB.fig.subplots_adjust(hspace=0.1, wspace=0.002)

    # set plotting function
    plotfun = my_plot
    dataset = "platelet-em"
    sample_idx = 5
    cmap = 'summer'
    print()

    for j, loss_function in enumerate(LOSS_FUNTION_ORDER):
        path = os.path.join("./weights/", dataset, "registration", loss_function)
        if not os.path.isdir(path):
            continue
        # load model
        checkpoint_path = os.path.join(path, "weights.ckpt")
        model = RegistrationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
        print(j, f'loss_function: {loss_function}')
        seg = '{seg}'
        ae = '{ae}'

        # run model
        I_0, S_0, I_m, S_m, I_1, S_1, inv_flow = get_img(model, sample_idx)
        crop_area = (270, 470, 100, 300)
        highlight_area = (350, 410, 102, 160)
        I_0 = crop(*crop_area, I_0)
        I_1 = crop(*crop_area, I_1)
        I_m = crop(*crop_area, I_m)
                    
        lossy = get_loss(dataset, loss_function, I_0, S_0, I_m, S_m, I_1, S_1)

        if 'deepsim' in loss_function:
            for depth, l in enumerate(lossy):
                # pre-processing
                lo = -l + 1
                lo = torch.log(1 + lo)
                lo -= lo.min()
                lo /= lo.max()
                print(f'min: {lo.min().item()}, max: {lo.max().item()}, mean: {lo.mean().item()}')
                vmin, vmax = lo.min().item(), lo.max().item()
                print(f'Size: {l.size()}')

                if loss_function == 'deepsim':
                    plotfun(figB, 0,j +(depth), model, lo.permute(0,2,1), 
                            title=f"$DeepSim^{depth}_{seg}$", cmap=cmap, 
                            vmin=vmin, vmax=vmax, interpolation='none')

                else:
                    plotfun(figB, 0,j-2 + depth, model, lo.permute(0,2,1), 
                            title=f"$DeepSim^{depth}_{ae}$", cmap=cmap, 
                            vmin=vmin, vmax=vmax, interpolation='none')
        else:
            print(f'Size: {lossy.size()}')
            if loss_function =="l2":
                loss = torch.log(1 + lossy)
                loss -= loss.min()
                loss /= loss.max()
                print(f'min: {loss.min()}, max: {loss.max()}, mean: {loss.mean()}')
                vmin, vmax = loss.min().item(), loss.max().item()
                vmax = loss.max().item() / 2
            else:
                loss = -lossy + 1
                loss = torch.log(1 + loss)
                loss -= loss.min()
                loss /= loss.max()
                print(f'min: {loss.min()}, max: {loss.max()}, mean: {loss.mean()}') 
                vmin, vmax = loss.min().item(), loss.max().item()
        print()

    os.makedirs("./out/plots/pdf/", exist_ok=True)
    os.makedirs("./out/plots/png/", exist_ok=True)
    figB.save("./out/plots/pdf/loss_sampleB.pdf", close=False)
    figB.save("./out/plots/png/loss_sampleB.png")

make_loss_A()
make_loss_B()