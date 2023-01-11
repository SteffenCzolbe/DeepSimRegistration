import torchreg
import torchreg.transforms.functional as f
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List
import nibabel as nib


class NiiDataset(Dataset):
    def __init__(
        self,
        image_nii_files: List[str],
        image_nii_label_files: List[str],
        min_intensity: float,
        max_intensity: float,
        quantile_normalization:bool=False,
    ):
        super().__init__()
        self.fnames = list(zip(image_nii_files, image_nii_label_files))
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.quantile_normalization = quantile_normalization

    def normalize_intensity(self, x):
        if not self.quantile_normalization:
            # normalize with given values
            return (x - self.min_intensity) / self.max_intensity
        else:
            # normalize with quantiles
            numpy_array = x.numpy()
            min_intensity, max_intensity = np.quantile(numpy_array, (0.01, 0.99))
            return torch.clamp((x - min_intensity) / max_intensity, 0, 1)

    def denormalize_intensity(self, x):
        if not self.quantile_normalization:
            return (x * self.max_intensity) + self.min_intensity
        else:
            return x*256

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        torchreg.settings.set_ndims(3)
        imgf, segf = self.fnames[idx]
        img = self.normalize_intensity(self.load_nii(imgf, dtype=torch.float))
        seg = self.load_nii(segf, dtype=torch.long)
        return img, seg

    def load_nii(self, path, dtype=torch.float):
        tensor = f.load_nii_as_tensor(path, dtype=dtype)
        return tensor


class NiiAtlasDataset(NiiDataset):
    def __init__(
        self,
        atlas_nii_file: str,
        atlas_nii_label_file: str,
        image_nii_files: List[str],
        image_nii_label_files: List[str],
        min_intensity: float,
        max_intensity: float,
        quantile_normalization:bool=False,
    ):
        super().__init__(
            image_nii_files, image_nii_label_files, min_intensity, max_intensity, quantile_normalization
        )
        self.atlas = self.normalize_intensity(self.load_nii(atlas_nii_file))
        self.atlas_seg = self.load_nii(atlas_nii_label_file, dtype=torch.long)

    def __getitem__(self, idx):
        img, seg = super().__getitem__(idx)
        return (img, seg), (self.atlas, self.atlas_seg)

class NiiSub2SubDataset(NiiDataset):
    def __init__(
        self,
        image_nii_files: List[str],
        image_nii_label_files: List[str],
        min_intensity: float,
        max_intensity: float,
        pairings_mode, #Literal['cross', 'pairs'],
        quantile_normalization:bool=False,
    ):
        super().__init__(
            image_nii_files, image_nii_label_files, min_intensity, max_intensity, quantile_normalization
        )
        self.n = len(self.fnames)
        self.pairings_mode = pairings_mode
        
    def __len__(self):
        if self.pairings_mode == 'cross':
            return self.n*(self.n-1)
        elif self.pairings_mode == 'pairs':
            return self.n
            

    def __getitem__(self, idx):
        if self.pairings_mode == 'cross':
            idx1 = idx // self.n
            idx2 = idx % self.n
        elif self.pairings_mode == 'pairs':
            idx1 = idx
            idx2 = self.n - idx - 1
        
        img1, seg1 = super().__getitem__(idx1)
        img2, seg2 = super().__getitem__(idx2)
        return (img1, seg1), (img2, seg2)
    