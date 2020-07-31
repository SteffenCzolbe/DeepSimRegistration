"""
Fine-grained transforms of 3D images

"""

import torch
import numpy as np
import torchreg.settings as settings
import nibabel as nib


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

def load_nii_as_tensor(path, dtype=torch.float, return_affine=False):
    """
    loads a tensor from an nii file.

    Parameters:
        path: file path to load
        dtype: torch datatype
        return_affine: set to True to additionally return the affine world to vox matrix. Default False

    Return:
        Torch Tensor, C x H x W x D
    """
    nii = nib.load(path)
    array = nii.get_fdata()
    tensor = volumetric_image_to_tensor(array, dtype=dtype).contiguous()
    affine = nii.affine
    return (tensor, affine) if return_affine else tensor

def save_tensor_as_nii(path, tensor, affine=None, dtype=np.float):
    """
    saves a tensor to an nii file.

    Parameters:
        path: file path to save
        tensor: Image volume, C x H x D x W
        affine: world to vox matrix. If none, the identity is chosen
        dtype: numpy dtype to cast to

    Return:
        Torch Tensor, C x H x W x D
    """
    if affine is None:
        affine = np.eye(4)
    else:
        affine = np.array(affine)

    array = image_to_numpy(tensor).astype(dtype)

    nii = nib.Nifti1Image(array, affine=affine)
    nib.save(nii, path)
    return
    