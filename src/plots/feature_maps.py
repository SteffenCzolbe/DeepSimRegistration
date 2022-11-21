import pickle
import os
import numpy as np
from tqdm import tqdm
from src.registration_model import RegistrationModel
import torch
import torchreg
import torchreg.viz as viz
from .config import *


def get_feature_maps(model, test_set_index):
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
        
        feats = model.deepsim.seg_model.extract_features(I_0)
        
    return [I_0, *feats]


def crop(x_low, x_high, y_low, y_high, I):
    return I[:, :, x_low:x_high, y_low:y_high]


def save_feature_map_platelet(save_path, feat, fullsize):
    crop_area_fullsize = (370, 650, 450, 730)
    this_size = feat.shape[-2:]
    shrink_factor = fullsize[0] / this_size[0]
    crop_area = [int(l // shrink_factor) for l in crop_area_fullsize] 
    feat = crop(*crop_area, feat)
    
    # remove batch
    feat = feat[0]
    
    # iterate over channels
    for i, feat_map in enumerate(feat):
        viz.export_img_2d(os.path.join(save_path, f'{i}.png'), feat_map.unsqueeze(0), normalize=True)
        

def save_feature_map_brain_mri(save_path, feat, fullsize):
    # remove batch
    D = feat.shape[3]
    feat = feat[0, :, :, int(D*0.42), :]
    
    # iterate over channels
    for i, feat_map in enumerate(feat):
        viz.export_img_2d(os.path.join(save_path, f'{i}.png'), feat_map.unsqueeze(0), normalize=True)
    


datasets = ["platelet-em", "brain-mri"]
loss_function = "deepsim"
sample_idx = 10

for dataset in datasets:
    if dataset == 'platelet-em':
        save_function = save_feature_map_platelet
    elif dataset == 'brain-mri':
        save_function = save_feature_map_brain_mri
        
    model_path = os.path.join("./weights/", dataset, "registration", loss_function)

    # load model
    checkpoint_path = os.path.join(model_path, "weights.ckpt")
    model = RegistrationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # run model
    feats = get_feature_maps(model, sample_idx)

    fullsize = feats[0].shape[-2:]


    save_path = f"./out/plots/{dataset}-featuremaps/"
    os.makedirs(save_path, exist_ok=True)

    # plot aligned image
    for i, feat in enumerate(feats):
        layer_path = os.path.join(save_path, f'{i}')
        os.makedirs(layer_path, exist_ok=True)
        save_function(layer_path, feat, fullsize)

