import torchreg
import torchreg.transforms as transforms
import numpy as np
import os
import nibabel as nib
from PIL import Image
    


def load_nii(path):
    # load fixed image
    nii = nib.load(path)
    array = nii.get_data()
    batch = transforms.functional.volumetric_image_to_tensor(array).unsqueeze(0)
    return nii, batch

def save_nii(path, nii, batch):
    array = transforms.functional.image_to_numpy(batch[0])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nii = nib.Nifti1Image(array, affine=nii.affine, header=nii.header)
    nib.save(nii, path)

def load_png(path):
    return transforms.functional.volumetric_image_to_tensor(np.array(Image.open(path))).unsqueeze(0)

def save_png(path, batch):
    array = transforms.functional.image_to_numpy(batch[0])
    Image.fromarray(array.astype(np.uint8)).save(path)

if __name__ == '__main__':
    """
    run with
    python3 -m torchreg.transforms.augmentation_tests.augmentation_test
    """
    # 2D
    torchreg.settings.set_ndims(2)

    # load image
    batch = load_png('./torchreg/transforms/augmentation_tests/pre-transform/test.png')

    # augment
    diffeomorphic_augment = transforms.RandomDiffeomorphic(p=1, m=2.5, r=16)
    diffeomorphic_augment.randomize(size=batch.shape[2:])
    batch_diffeomorphic = diffeomorphic_augment(batch)

    affine_augment = transforms.RandomAffine(degrees=(-180, 180), translate=(-1, 1), scale=(0.9, 1.1), shear=(-0.03, 0.03), flip=True)
    affine_augment.randomize()
    batch_affine = affine_augment(batch)

    # save augmented images
    save_png('./torchreg/transforms/augmentation_tests/post-transform/diffeomorphic.png', batch_diffeomorphic)
    save_png('./torchreg/transforms/augmentation_tests/post-transform/affine.png', batch_affine)
    

    # 3D
    torchreg.settings.set_ndims(3)

    # load image
    nii, batch = load_nii('./torchreg/transforms/augmentation_tests/pre-transform/brain.nii.gz')

    # augment
    diffeomorphic_augment = transforms.RandomDiffeomorphic(p=1, m=2.5, r=16)
    diffeomorphic_augment.randomize(size=batch.shape[2:])
    batch_diffeomorphic = diffeomorphic_augment(batch)

    affine_augment = transforms.RandomAffine(degrees=(-180, 180), translate=(-1, 1), scale=(0.9, 1.1), shear=(-0.03, 0.03), flip=True)
    affine_augment.randomize()
    batch_affine = affine_augment(batch)

    # save augmented images
    save_nii('./torchreg/transforms/augmentation_tests/post-transform/diffeomorphic.nii.gz', nii, batch_diffeomorphic)
    save_nii('./torchreg/transforms/augmentation_tests/post-transform/affine.nii.gz', nii, batch_affine)

