"""
Transforms to work on tuples of annotated images. 
Multiple transforms can be combined with torchvision.transforms.Compose
"""

import torch
import numpy as np
from .functional import *
import torchreg.settings as settings


class ToTensor:
    """
    transforms an numpy array to tensor
    """

    def __call__(self, array, dtype=torch.float):
        return volumetric_image_to_tensor(array, dtype=dtype)


class ToNumpy:
    """
    transforms an tensor to numpy array
    """

    def __call__(self, tensor):
        return image_to_numpy(tensor)
