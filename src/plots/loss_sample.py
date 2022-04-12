import pickle
import os
import numpy as np
from tqdm import tqdm
from src.test_registration_voxelmorph import RegistrationModel
import torch
import torchreg
import torchreg.viz as viz
from .config import *
import matplotlib.patches as patches

from src.segmentation_model import SegmentationModel
from src.autoencoder_model import AutoEncoderModel
from src.loss_metrics import NCC, DeepSim, VGGFeatureExtractor, NMI, MIND_loss
import torch.nn.functional as F
import os
from scipy.stats.stats import pearsonr

LOSS_FUNTION_ORDER.remove('mind')
LOSS_FUNTION_ORDER.remove("nmi")
LOSS_FUNTION_ORDER.remove("ncc2+supervised")
  
def get_img(model, test_set_index):
    transformer = torchreg.nn.SpatialTransformer()
    integrate = torchreg.nn.FlowIntegration(6)

    with torch.no_grad():
        #device = "cuda" if torch.cuda.is_available() else "cpu"
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

def my_plot(fig, row, col, model, I, title, cmap='gray', vmin=0, vmax=1,interpolation='none'):
    # crop_area = (350, 410, 100, 160)
    # highlight_area = (350, 409, 100, 159)
    fig.plot_img(row, col, I[0], cmap=cmap, title=title, vmin=vmin, vmax=vmax,interpolation=interpolation)
    # h = highlight_area[1] - highlight_area[0]
    # w = highlight_area[3] - highlight_area[2]
    # p = (highlight_area[2] - crop_area[2],
    #         highlight_area[0] - crop_area[0])
    # rect = patches.Rectangle(p, h, w, linewidth=4,
    #                             edgecolor='blue', facecolor='none')
    # fig.axs[row, col].add_patch(rect)



def make_loss():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7})

    #fig = viz.Fig(1, len(LOSS_FUNTION_ORDER) + 6, None, figsize=(9, 2))
    fig = viz.Fig(2, len(LOSS_FUNTION_ORDER) + 2, None, figsize=(6, 2))
    fig.fig.subplots_adjust(hspace=0.4, wspace=0.004)

    # set plotting function
    plotfun = my_plot
    dataset = "platelet-em"
    sample_idx = 5

    for j, loss_function in enumerate(LOSS_FUNTION_ORDER):
        path = os.path.join("./weights/", dataset,
                            "registration", loss_function)
        if not os.path.isdir(path):
            continue
        # load model
        checkpoint_path = os.path.join(path, "weights.ckpt")
        model = RegistrationModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path)
        print(j,loss_function)
        seg = '{seg}'
        ae = '{ae}'

        # run model
        I_0, S_0, I_m, S_m, I_1, S_1, inv_flow = get_img(model, sample_idx)
        # I_1 = I_1.permute(0,1,3,2)
        # I_0 = I_0.permute(0,1,3,2)
        crop_area = (270, 470, 100, 300)
        highlight_area = (350, 410, 102, 160)
        I_0 = crop(*crop_area, I_0)
        I_1 = crop(*crop_area, I_1)
        I_m = crop(*crop_area, I_m)

        plotfun(fig, 0, 0, model, I_m ,title='Moved',
                    vmin=0, vmax=1)

        plotfun(fig, 0, 1, model, I_1 ,title='Fixed',
                    vmin=0, vmax=1)
                    
        lossy = get_loss(dataset, loss_function, I_0, S_0, I_m, S_m, I_1, S_1)
        if 'deepsim' in loss_function:
            seg_outputs = []
            ae_outputs = []
            for depth, l in enumerate(lossy):

                # pre-processing
                lo = -l + 1
                lo = torch.log(1 + lo)
                # lo -= lo.min()
                # lo /= lo.max()
                print(lo.min(), lo.max(), lo.mean())
                vmin, vmax = 0, lo.max().item()
                #vmax = 1

                print(l.size())
                if loss_function == 'deepsim':
                    seg_out = F.interpolate(lo.unsqueeze(0), size=(200, 200), mode='bilinear', align_corners=False).squeeze(0)
                    seg_outputs.append(seg_out)

                    # plotfun(fig,0,j+2 +(depth+2),model,lo,title=f"$DeepSim^{depth}_{seg}$",cmap='PiYG', vmin=vmin, vmax=vmax,
                    # interpolation='bilinear')
                    plotfun(fig,1,j +(depth),model,lo.permute(0,2,1),title=f"$DeepSim^{depth}_{seg}$",cmap='PiYG', vmin=vmin, vmax=vmax,
                    interpolation='none')

                else:
                    ae_out = F.interpolate(lo.unsqueeze(0), size=(200, 200), mode='bilinear', align_corners=False).squeeze(0)
                    ae_outputs.append(ae_out)
                    # plotfun(fig,0,j+2 +(depth),model,lo,title=f"$DeepSim^{depth}_{ae}$",cmap='PiYG', vmin=vmin, vmax=vmax,
                    # interpolation='bilinear')
                    plotfun(fig,1,j-2 + depth,model,lo.permute(0,2,1),title=f"$DeepSim^{depth}_{ae}$",cmap='PiYG', vmin=vmin, vmax=vmax,
                    interpolation='none')
            if loss_function == 'deepsim':
                seg_lvl = torch.stack(seg_outputs)
                mean_seg_lvl = torch.mean(seg_lvl, dim = 0)
            else:
                ae_lvl = torch.stack(ae_outputs)
                mean_ae_lvl = torch.mean(ae_lvl, dim = 0)

            
            
            
        
        else:
            print(lossy.size())
            if loss_function =="l2":
                loss = torch.log(1 + lossy)
                # loss -= loss.min()
                # loss /= loss.max()
                print(loss.min(), loss.max(), loss.mean())
                vmin, vmax = 0, loss.max().item()
                vmax = 0.2
            else:
                loss = -lossy + 1
                loss = torch.log(1 + loss)
                # loss -= loss.min()
                # loss /= loss.max()
                print(loss.min(), loss.max(), loss.mean())  
                vmin, vmax = 0, loss.max().item()
                #vmax = 1
            plotfun(fig,0, j+2, model, loss ,title=LOSS_FUNTION_CONFIG[loss_function]["display_name"],
                    vmin=vmin, vmax=vmax,cmap='PiYG')
        print()

    

    print(mean_seg_lvl.size(), mean_ae_lvl.size())
    print(mean_seg_lvl.min(), mean_seg_lvl.max(), mean_seg_lvl.mean())
    fig.plot_img(0, 4, mean_ae_lvl, cmap='PiYG', title=f"$DeepSim_{ae}$", vmin=0, vmax=1,interpolation='bilinear')
    fig.plot_img(0, 5, mean_seg_lvl, cmap='PiYG', title=f"$DeepSim_{seg}$", vmin=0, vmax=1,interpolation='bilinear')
    seg_np = mean_seg_lvl.numpy().reshape(-1)
    ae_np = mean_ae_lvl.numpy().reshape(-1)
    print(ae_np.shape)

    corr = np.corrcoef(seg_np, ae_np)
    print(corr)

    print(pearsonr(seg_np, ae_np))


    os.makedirs("./out/plots", exist_ok=True)
    fig.save("./out/plots/loss_sample.pdf", close=False)
    fig.save("./out/plots/loss_sample.png")


#make_overview()
#make_detail_all()
make_loss()