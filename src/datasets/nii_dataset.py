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
        max_intensity: float
    ):
        super().__init__()
        self.fnames = list(zip(image_nii_files, image_nii_label_files))
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def normalize_intensity(self, x):
        return (x - self.min_intensity) / self.max_intensity

    def denormalize_intensity(self, x):
        return (x * self.max_intensity) + self.min_intensity

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
        max_intensity: float
    ):
        super().__init__(image_nii_files, image_nii_label_files, min_intensity, max_intensity)
        self.atlas = self.normalize_intensity(self.load_nii(atlas_nii_file))
        self.atlas_seg = self.load_nii(atlas_nii_label_file, dtype=np.int)
    
    def __getitem__(self, idx):
        img, seg = super().__getitem__(idx)
        return (img, seg), (self.atlas, self.atlas_seg)