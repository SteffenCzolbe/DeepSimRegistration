import os
import glob
import pandas as pd
from .nii_dataset import NiiDataset, NiiAtlasDataset
import torchreg.transforms.functional as f

def HeartMRIDataset(path: str, split: str, pairs=False):
    """
    creates the HeartMRI dataset.

    Parameters:
        path: path to the dataset folder
        split: the dataset split. eg: 'train'
        pairs: bool, if True return pairs for registration
    """

    if pairs:
        raise NotImplementedException('Heart-MRI dataset is not for registration.')
    if split == 'test':
        print('WARNING: Heart-MRI dataset has no test set. Returning validation set instead.')
        split = 'val'

    subjects = sorted(glob.glob(os.path.join(path, 'processed', '*')))
    image_nii_files = list(map(lambda s: os.path.join(s, 'axial.nii.gz'), subjects))
    image_nii_label_files = list(map(lambda s: os.path.join(s, 'label_B.nii.gz'), subjects))

    min_intensity = 150
    max_intensity = 256

    if split == 'train':
        return NiiDataset(image_nii_files[:-2], image_nii_label_files[:-2], min_intensity, max_intensity)
    elif split=='val':
        return NiiDataset(image_nii_files[-2:], image_nii_label_files[-2:], min_intensity, max_intensity)


