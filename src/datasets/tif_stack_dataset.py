import torchreg
import torchreg.transforms as transforms
import torchreg.transforms.functional as f
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import numpy as np


class TiffStackDataset(Dataset):
    def __init__(
        self,
        intensity_tif_image,
        segmentation_tif_image,
        min_slice=0,
        max_slice=-1,
        slice_pairs=False,
        slice_pair_max_z_diff=(2, 2),
        transform=None,
    ):
        """
        Creates a dataset of the images of a Tif(f) image stack.
        Parameters:
            intensity_tif_image: path to intensity image
            segmentation_tif_image: path to segmentation image
            min_slice: lowest slice of the stack to test. Default 0.
            max_slice: highest slice of the stack to take. Exclusive. Default -1.
            slice_pairs: bool, should a pair of slices be returned? Default False
            slice_pair_max_z_diff: max z-diff beween slice pairs
            transform: Data-augmentation transforms
        """
        self.intensity_stack = Image.open(intensity_tif_image)
        self.segmentation_stack = Image.open(segmentation_tif_image)
        self.min_slice = min_slice
        self.max_slice = self.intensity_stack.n_frames if max_slice == -1 else max_slice
        self.dynamic_range = self.get_dynamic_range()
        self.slice_pairs = slice_pairs
        self.max_z_diff = slice_pair_max_z_diff
        self.transform = transform

    def get_dynamic_range(self):
        mode = self.intensity_stack.mode
        if "16" in mode:
            return 16
        elif "32" in mode:
            return 32
        else:
            return 8

    def toIntTensor(self, x):
        x = np.array(x, dtype=np.int8)
        return f.volumetric_image_to_tensor(x, dtype=torch.long)

    def toFloatTensor(self, x):
        x = np.array(x, dtype=np.float32)
        return f.volumetric_image_to_tensor(x, dtype=torch.float32)

    def normalize_intensity(self, x):
        return x / 2 ** self.dynamic_range

    def denormalize_intensity(self, x):
        return x * 2 ** self.dynamic_range

    def __len__(self):
        if self.slice_pairs and self.max_z_diff == (0, 1):
            # last slice has no next partner, thus we reduce the length by one
            return self.max_slice - self.min_slice - 1
        return self.max_slice - self.min_slice

    def __getitem__(self, idx):
        torchreg.settings.set_ndims(2)
        idx0 = idx + self.min_slice
        self.intensity_stack.seek(idx0)
        self.segmentation_stack.seek(idx0)
        img0, seg0 = self.intensity_stack.copy(), self.segmentation_stack.copy()

        if not self.slice_pairs:
            if self.transform:
                img0, seg0 = self.transform(img0, seg0)
            return (
                self.normalize_intensity(self.toFloatTensor(img0)),
                self.toIntTensor(seg0),
            )

        # pick second slice
        idx1 = random.choice(
            list(range(max(self.min_slice, idx0 - self.max_z_diff[0]), idx0))
            + list(range(idx0 + 1, min(self.max_slice, idx0 + self.max_z_diff[1] + 1)))
        )
        self.intensity_stack.seek(idx1)
        self.segmentation_stack.seek(idx1)
        img1, seg1 = self.intensity_stack.copy(), self.segmentation_stack.copy()
        if self.transform:
            img0, seg0, img1, seg1 = self.transform(img0, seg0, img1, seg1)
        return (
            (
                self.normalize_intensity(self.toFloatTensor(img0)),
                self.toIntTensor(seg0),
            ),
            (
                self.normalize_intensity(self.toFloatTensor(img1)),
                self.toIntTensor(seg1),
            ),
        )


if __name__ == "__main__":
    transform = transforms.StackCompose(
        [
            transforms.StackRandomHorizontalFlip(),
            transforms.StackRandomVerticalFlip(),
            transforms.StackRandomAffine(
                degrees=180, scale=(0.8, 1.2), shear=20, fillcolor=0
            ),
        ]
    )
    d = PlantEMDataset(
        "./data/platelet_em/images/24-images.tif",
        "./data/platelet_em/labels-class/24-class.tif",
        max_slice=1,
        slice_pairs=True,
        slice_pair_max_z_diff=1,
        transform=transform,
    )

    import matplotlib.pyplot as plt

    (i0, s0), (i1, s1) = d[0]
    plt.imshow(s0)
    plt.show()
    plt.imshow(s1)
    plt.show()
