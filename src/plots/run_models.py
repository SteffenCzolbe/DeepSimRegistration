from argparse import ArgumentParser
import numpy as np
import os
import pickle
from tqdm import tqdm
import torch

from src.registration_model import RegistrationModel
from src.test_registration_voxelmorph import RegistrationModel as RegistrationModelOLD
from .config2D import *

#os.environ["CUDA_LAUNCH_BLOCKING"]='1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='5'
print(os.environ["CUDA_VISIBLE_DEVICES"])

def test_model(model):
    def map_dicts(list_of_dics):
        dict_of_lists = {}
        for k in list_of_dics[0].keys():
            dict_of_lists[k] = torch.stack(
                [d[k] for d in list_of_dics]).cpu().numpy()
        return dict_of_lists

    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        model = model.to(device)
        test_set = model.test_dataloader().dataset

        scores = []
        for i in range(len(test_set)):
            (I_0, S_0), (I_1, S_1) = test_set[i]
            batch = (
                (I_0.unsqueeze(0).to(device), S_0.unsqueeze(0).to(device)),
                (I_1.unsqueeze(0).to(device), S_1.unsqueeze(0).to(device)),
            )
            score = model._step(batch, None, eval_per_class=True)
            scores.append(score)
        # map list of dicts to a dict of lists
        return map_dicts(scores)


def run_models(use_cached=True, model='voxelmorph'):
    if model == 'voxelmorph':
        # load results for 3d brain-mri dataset
        cache_file_name_3D = "./src/plots/cache.pickl"
        cache_file_name_3D_mind = "./src/plots/cache3D_mind.pickl"
        # load results for 2d datasets
        cache_file_name_2D = "./src/plots/cache2D_mind.pickl"
    elif model == 'transmorph':
        # load results for 2d datasets
        cache_file_name_2D = "./src/plots/cache2D_transmorph.pickl"

    if use_cached and os.path.isfile(cache_file_name_2D):
        if model == 'voxelmorph':
            d1 = pickle.load(open(cache_file_name_3D, "rb"))
            d1b = pickle.load(open(cache_file_name_3D_mind, "rb"))
            d1['brain-mri'].update({'mind': d1b['brain-mri']['mind']})
            del d1['phc-u373']
            del d1['platelet-em']

        d2 = pickle.load(open(cache_file_name_2D, "rb"))

        if model == 'voxelmorph':        
            d = {**d1, **d2}
        elif model == 'transmorph':
            d = d2
        else:
            raise ValueError(f'model "{args.net}" unknow. \n use voxelmorph or transmorph')

        return d
        
    else: 
        if os.path.isfile(cache_file_name_2D):
            results = pickle.load(open(cache_file_name_2D, "rb"))
        else:
            results = {}

        if model == 'voxelmorph':
            folder = 'registration'
        elif model == 'transmorph':
            folder = 'transmorph'
        else:
            raise ValueError(f'model "{args.net}" unknow. \n use voxelmorph or transmorph') 
        
        # run models for 2d datasets (config2D) - change to config to run for both 2d and 3d datasets
        for dataset in DATASET_ORDER:
            results[dataset] = {}
            for loss_function in tqdm(
                LOSS_FUNTION_ORDER, desc=f"testing loss-functions on {dataset}"
            ):
                path = os.path.join(
                    "./weights/", dataset, folder, loss_function
                )
                if not os.path.isdir(path):
                    continue
                # load model
                checkpoint_path = os.path.join(path, "weights.ckpt")

                if model == 'voxelmorph':
                    # hard-coded self.hparams.net for old voxelmoprh checkpoints!
                    registration_model = RegistrationModelOLD.load_from_checkpoint(
                        checkpoint_path=checkpoint_path
                    )
                else:
                    registration_model = RegistrationModel.load_from_checkpoint(
                        checkpoint_path=checkpoint_path
                    )
                #print(loss_function, checkpoint_path)
                step_dict = test_model(registration_model)
                results[dataset][loss_function] = step_dict
                #print(f"{dataset}, {loss_function}:")
                #print(step_dict)
        pickle.dump(results, open(cache_file_name_2D, "wb"))
        return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--net', type=str, default='voxelmorph', help='voxelmorph or transmorph.')
    args = parser.parse_args()
    run_models(use_cached=False, model=args.net)
