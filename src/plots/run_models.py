import pickle
import os
import numpy as np
from tqdm import tqdm
from src.registration_model import RegistrationModel
import torch
from .config import *


def test_model(model):
    def map_dicts(list_of_dics):
        dict_of_lists = {}
        for k in list_of_dics[0].keys():
            dict_of_lists[k] = torch.stack([d[k] for d in list_of_dics]).cpu().numpy()
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
    return map_dicts(scores)


def run_models(use_cached=True):
    cache_file_name = "./src/plots/cache.pickl"
    if use_cached and os.path.isfile(cache_file_name):
        return pickle.load(open(cache_file_name, "rb"))
    else:
        results = {}
        #results = pickle.load(open(cache_file_name, "rb"))
        for dataset in DATASET_ORDER:
            results[dataset] = {}
            for loss_function in tqdm(
                LOSS_FUNTION_ORDER, desc=f"testing loss-functions on {dataset}"
            ):
                path = os.path.join(
                    "./weights/", dataset, "registration", loss_function
                )
                if not os.path.isdir(path):
                    continue
                # load model
                checkpoint_path = os.path.join(path, "weights.ckpt")
                model = RegistrationModel.load_from_checkpoint(
                    checkpoint_path=checkpoint_path
                )
                step_dict = test_model(model)
                results[dataset][loss_function] = step_dict
        pickle.dump(results, open(cache_file_name, "wb"))
        return results


if __name__ == "__main__":
    run_models(use_cached=False)
