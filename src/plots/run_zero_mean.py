import numpy as np
import os
import pickle
from tqdm import tqdm
import torch

from src.registration_model import RegistrationModel
from .config2D import *

# #os.environ["CUDA_LAUNCH_BLOCKING"]='1'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]='3'
# print(os.environ["CUDA_VISIBLE_DEVICES"])

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


def run_models(use_cached=True):

    cache_file_name = "./src/plots/cache2D_zero_mean.pickl"
    
    if use_cached and os.path.isfile(cache_file_name):
        return pickle.load(open(cache_file_name, "rb"))
    else:
        if os.path.isfile(cache_file_name):
            results = pickle.load(open(cache_file_name, "rb"))
        else:
            results = {}

        for dataset in DATASET_ORDER:
            print(dataset)
            results[dataset] = {}
            for loss_function in tqdm(
                EXTRACT_ZERO_MEAN_LOSS_FUNCTIONS, desc=f"testing loss-functions on {dataset}"
            ):
                path = os.path.join(
                    "./weights_experiments/", "z-extras/zero-mean", dataset, loss_function
                )
                print(path)
                if not os.path.isdir(path):
                    print('YO')
                    continue
                # load model
                checkpoint_path = os.path.join(path, "weights.ckpt")
                #print(checkpoint_path)
                model = RegistrationModel.load_from_checkpoint(
                    checkpoint_path=checkpoint_path
                )
                step_dict = test_model(model)
                results[dataset][loss_function] = step_dict
                #print(f"{dataset}, {loss_function}:")
                #print(step_dict)
        pickle.dump(results, open(cache_file_name, "wb"))
        return results


if __name__ == "__main__":
    run_models(use_cached=False)