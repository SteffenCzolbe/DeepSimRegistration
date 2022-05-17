"""
functions providing dimension-agnostic contructors to popular torch.nn building blocks
"""
import torch.nn as nn
import torchreg.settings as settings


def Conv(*args, **kwargs):
    ndims = settings.get_ndims()
    if ndims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif ndims == 3:
        return nn.Conv3d(*args, **kwargs)
    else:
        raise Exception()


def Upsample(size=None, scale_factor=None, mode="nearest", align_corners=False):
    ndims = settings.get_ndims()
    mode = interpol_mode(mode)

    if ndims == 2:
        return nn.Upsample(
            size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners
        )
    elif ndims == 3:
        return nn.Upsample(
            size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners
        )
    else:
        raise Exception()


def BatchNorm(*args, **kwargs):
    ndims = settings.get_ndims()
    if ndims == 2:
        return nn.BatchNorm2d(*args, **kwargs)
    elif ndims == 3:
        return nn.BatchNorm3d(*args, **kwargs)
    else:
        raise Exception()


def Dropout(*args, **kwargs):
    """
    performs channel-whise dropout. As described in the paper Efficient Object Localization Using Convolutional 
    Networks , if adjacent pixels within feature maps are strongly correlated (as is normally the case in early 
    convolution layers) then i.i.d. dropout will not regularize the activations and will otherwise just result 
    in an effective learning rate decrease.
    """
    ndims = settings.get_ndims()
    if ndims == 2:
        return nn.Dropout2d(*args, **kwargs)
    elif ndims == 3:
        return nn.Dropout3d(*args, **kwargs)
    else:
        raise Exception()


def interpol_mode(mode):
    """
    returns an interpolation mode for the current dimensioanlity.
    """
    ndims = settings.get_ndims()
    if mode in ["linear", "bilinear", "bicubic", "trilinear"]:
        mode = ["linear", "bilinear", "trilinear"][ndims - 1]
    return mode
