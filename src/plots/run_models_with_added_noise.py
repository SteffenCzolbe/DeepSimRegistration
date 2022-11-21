import pickle
import os
import numpy as np
from tqdm import tqdm
from src.registration_model import RegistrationModel
import torch
from .config import *
import torchvision

def add_gaussian_noise(img, std):
    noise_dist = torch.distributions.normal.Normal(0, std)
    noise = noise_dist.sample(img.shape).to(img.device)
    return torch.clamp(img + noise, 0, 1)

def add_gaussian_smoothing(img, std):
    def gkern2d(l=5, sig=1.):
        """
        creates a 2d gaussian kernel with side length `l` and a sigma of `sig`
        """
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)
    k_size = max(5, int((std*7)//2)*2 + 1) 
    blurring_kernel =  gkern2d(k_size, std)
    blurring_kernel = torch.tensor(blurring_kernel, dtype=img.dtype, device=img.device).view(1,1,k_size,k_size)
    blurred = torch.nn.functional.conv2d(img, blurring_kernel, padding=k_size//2)
    return blurred

def add_noise_or_smoothing(img, std):
    """
    adds gaussian noise with sigma=std.
    if std is negative, smooth with a gaussian kernel of that size instead.
    """
    if std >= 0:
        return add_gaussian_noise(img, std)
    else:
        return add_gaussian_smoothing(img, -std)
    
def save_img_as_png(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = torchvision.transforms.ToPILImage()(tensor[0].cpu())
    img.save(path)
    

def test_model_with_noise(model, std):
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
            I_0 = I_0.unsqueeze(0).to(device)
            I_0 = add_noise_or_smoothing(I_0, std)
            S_0 = S_0.unsqueeze(0).to(device)
            I_1 = I_1.unsqueeze(0).to(device)
            I_1 = add_noise_or_smoothing(I_1, std)
            S_1 = S_1.unsqueeze(0).to(device)
            batch = ((I_0, S_0),(I_1, S_1))
            score = model._step(batch, None, eval_per_class=True)
            scores.append(score)
            
            if i == 0:
                # save one example
                save_img_as_png(I_0, os.path.join('out', 'plots', 'png', 'noisy_images', f'std_{std}.png'))
            
            
        # map list of dicts to a dict of lists
        return map_dicts(scores)


def run_models(use_cached=True):
    cache_file_name = "./src/plots/cache_with_added_noise.pickl"

    if use_cached and os.path.isfile(cache_file_name):
        return pickle.load(open(cache_file_name, "rb"))
    else:
        if os.path.isfile(cache_file_name):
            results = pickle.load(open(cache_file_name, "rb"))
        else:
            results = {}

        dataset = "platelet-em"
        results[dataset] = {}
        for loss_function in tqdm(
            LOSS_FUNTION_ORDER, desc=f"testing loss-functions on {dataset}"
        ):
            results[dataset][loss_function] = {}
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
            
            for noise_std in [-2, -1, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
                step_dict = test_model_with_noise(model, noise_std)
                results[dataset][loss_function][noise_std] = step_dict
                
                # DEBUG
                #print(f"{dataset}, {loss_function}, noise {noise_std}:")
                #print({k:np.mean(v) for k,v in step_dict.items()})
        pickle.dump(results, open(cache_file_name, "wb"))
        return results


if __name__ == "__main__":
    run_models(use_cached=False)