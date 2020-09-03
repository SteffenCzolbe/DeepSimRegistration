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
    return transforms.functional.volumetric_image_to_tensor(
        np.array(Image.open(path))
    ).unsqueeze(0)


def save_png(path, batch):
    array = transforms.functional.image_to_numpy(batch[0])
    array = np.clip(array, 0, 255)
    Image.fromarray(array.astype(np.uint8)).save(path)


if __name__ == "__main__":
    """
    run with
    python3 -m torchreg.transforms.augmentation_tests.augmentation_test
    """
    # 2D
    torchreg.settings.set_ndims(2)

    # load image
    batch = load_png("./torchreg/transforms/augmentation_tests/pre-transform/test.png")

    # augment
    noise_augment = transforms.GaussianNoise(std=20)
    noise_augment.randomize()
    batch_noise = noise_augment(batch)

    intensity_augment = transforms.RandomIntenityShift(
        instensity_change_std=60, size_std=800, filter_count=15, downsize=8
    )
    intensity_augment.randomize()
    batch_intensity = intensity_augment(batch)

    # save augmented images
    save_png(
        "./torchreg/transforms/augmentation_tests/post-transform/noise.png", batch_noise
    )
    save_png(
        "./torchreg/transforms/augmentation_tests/post-transform/intensity.png",
        batch_intensity,
    )

    # 3D
    torchreg.settings.set_ndims(3)

    # load image
    nii, batch = load_nii(
        "./torchreg/transforms/augmentation_tests/pre-transform/brain.nii.gz"
    )

    # augment
    noise_augment = transforms.GaussianNoise(std=10)
    noise_augment.randomize()
    batch_noise = noise_augment(batch)

    intensity_augment = transforms.RandomIntenityShift(
        instensity_change_std=60, size_std=200, filter_count=20, downsize=8
    )
    intensity_augment.randomize()
    batch_intensity = intensity_augment(batch)

    # save augmented images
    save_nii(
        "./torchreg/transforms/augmentation_tests/post-transform/noise.nii.gz",
        nii,
        batch_noise,
    )
    save_nii(
        "./torchreg/transforms/augmentation_tests/post-transform/intensity.nii.gz",
        nii,
        batch_intensity,
    )

