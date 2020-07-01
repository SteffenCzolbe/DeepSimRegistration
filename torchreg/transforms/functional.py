"""
Fine-grained transforms of 3D images

"""

import torch
import numpy as np
import torchreg.settings as settings


def volumetric_image_to_tensor(img, dtype=torch.float32):
    """
    transforms a 3d np.array indexed as H x W x D x C to a torch.tensor indexed as C x H x W x D
    """
    ndims = settings.get_ndims()
    if len(img.shape) == ndims:
        # add channel-dim
        img = np.expand_dims(img, -1)
    # permute channel to front
    permuted = np.moveaxis(img, -1, 0)
    return torch.as_tensor(np.array(permuted), dtype=dtype)


def image_to_numpy(tensor):
    """
    transforms a 2d or 3d torch.tensor to a numpy array.
    C x H x W x D becomes H x W x D x C
    1 x H x W x D becomes H x W x D
    C x H x W becomes H x W x C
    1 x H x W becomes H x W
    """
    # to numpy
    array = tensor.detach().cpu().numpy()
    # channel to back
    permuted = np.moveaxis(array, 0, -1)
    # remove channel of size 1
    if permuted.shape[-1] == 1:
        permuted = permuted[..., 0]
    return permuted


def landmarks_to_tensor(landmarks):
    """
    transforms a 2d np.array indexed as N x C to a torch.tensor indexed as C x N
    """
    return torch.as_tensor(np.array(landmarks.T), dtype=torch.float32)


def landmarks_to_numpy(tensor):
    """
    transforms a 2d np.array indexed as N x C to a torch.tensor indexed as C x N
    """
    return tensor.detach().cpu().numpy().T
