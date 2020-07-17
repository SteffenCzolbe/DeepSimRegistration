import torchreg
import torchreg.transforms as transforms
import numpy as np
import os
import nibabel as nib
    


def load_nii(path):
    # load fixed image
    nii = nib.load(path)
    array = nii.get_data()
    return nii, array

def save_nii(path, nii, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nii = nib.Nifti1Image(array, affine=nii.affine, header=nii.header)
    nib.save(nii, path)

if __name__ == '__main__':
    """
    run with
    python3 -m torchreg.transforms.augmentation_tests.augmentation_test
    """
    torchreg.settings.set_ndims(3)

    # load image
    nii, img = load_nii('./torchreg/transforms/augmentation_tests/pre-transform/brain.nii.gz')
    batch = transforms.functional.volumetric_image_to_tensor(img).unsqueeze(0)

    # augment
    diffeomorphic_augment = transforms.RandomDiffeomorphic(p=1, m=2.5, r=16)
    diffeomorphic_augment.randomize(size=batch.shape[2:])
    batch_diffeomorphic = diffeomorphic_augment(batch)

    affine_augment = transforms.RandomAffine(degrees=(-180, 180), translate=(-1, 1), scale=(0.9, 1.1), shear=(-0.03, 0.03), flip=True)
    affine_augment.randomize()
    batch_affine = affine_augment(batch)

    # transform back to numpy
    diffeomorphic = transforms.functional.image_to_numpy(batch_diffeomorphic[0])
    affine = transforms.functional.image_to_numpy(batch_affine[0])

    # save augmented images
    save_nii('./torchreg/transforms/augmentation_tests/post-transform/diffeomorphic.nii.gz', nii, diffeomorphic)
    save_nii('./torchreg/transforms/augmentation_tests/post-transform/affine.nii.gz', nii, affine)