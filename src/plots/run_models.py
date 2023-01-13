from argparse import ArgumentParser
import numpy as np
import os
import pickle
from tqdm import tqdm
import torch

from src.registration_model import RegistrationModel
from .config import *

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
        for i in tqdm(range(len(test_set)), desc='testing samples...'):
            (I_0, S_0), (I_1, S_1) = test_set[i]
            batch = (
                (I_0.unsqueeze(0).to(device), S_0.unsqueeze(0).to(device)),
                (I_1.unsqueeze(0).to(device), S_1.unsqueeze(0).to(device)),
            )
            score = model._step(batch, None, eval_per_class=True)
            scores.append(score)
        # map list of dicts to a dict of lists
        return map_dicts(scores)


def run_models(use_cached=True, model='voxelmorph', dry_run=False):
    
    if model == 'voxelmorph':
        cache_file_name = "./src/plots/cache.pickl"
    elif model == 'transmorph':
        cache_file_name = "./src/plots/cache_transmorph.pickl"

    if os.path.isfile(cache_file_name):
        results = pickle.load(open(cache_file_name, "rb"))
    else:
        results = {}
        
    if use_cached:
        return results

    if model == 'voxelmorph':
        folder = 'registration'
    elif model == 'transmorph':
        folder = 'transmorph'
    else:
        raise ValueError(f'model "{args.net}" unknow. \n use voxelmorph or transmorph') 
    
    # run models
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
                registration_model = RegistrationModel.load_from_checkpoint(
                    checkpoint_path=checkpoint_path
                )
            else:
                registration_model = RegistrationModel.load_from_checkpoint(
                    checkpoint_path=checkpoint_path
                )
            print(f'loading model for {loss_function} from {checkpoint_path}.')
            if not dry_run:
                step_dict = test_model(registration_model)
                results[dataset][loss_function] = step_dict
                print(f"results for {dataset}, {loss_function}:")
                print(step_dict)
                
    if not dry_run:
        pickle.dump(results, open(cache_file_name, "wb"))
    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--net', type=str, default='voxelmorph', help='voxelmorph or transmorph.')
    parser.add_argument(
        '--dry_run', action='store_true', help='Set to try model loading.')
    args = parser.parse_args()
    run_models(use_cached=False, model=args.net, dry_run=args.dry_run)
