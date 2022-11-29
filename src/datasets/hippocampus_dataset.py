import os
import pandas as pd
from .nii_dataset import NiiDataset, NiiAtlasDataset, NiiSub2SubDataset
import torchreg.transforms.functional as f
import json
import random


def HippocampusMRDataset(path: str, split: str, pairs:bool):
    """
    creates the Hippocampus dataset.

    Parameters:
        path: path to the dataset folder. eg. "../HippocampusMR"
        split: the dataset split. eg: 'train'
        pairs: bool, if True return pairs for registration
    """

    # load subjects
    with open(os.path.join(path, 'HippocampusMR_dataset.json'), 'r') as f:
        dataset_config = json.load(f)
        
    subjects = dataset_config["training"]
    
    # make splits
    # total 260 subjects
    n = len(subjects)
    random.Random(42).shuffle(subjects)
    
    if split == 'train':
        subjects = subjects[:int(0.6*n)]
    elif split == 'val':
        subjects = subjects[int(0.6*n):int(0.8*n)]
    elif split == 'test':
        subjects = subjects[int(0.8*n):]
    
    image_nii_files = list(
        map(lambda s: os.path.join(path, s["image"]), subjects)
    )
    image_nii_label_files = list(
        map(lambda s: os.path.join(path, s["label"]), subjects)
    )

    if pairs:
        return NiiSub2SubDataset(image_nii_files, image_nii_label_files, None, None, quantile_normalization=True)
    else:
        return NiiDataset(
                image_nii_files, image_nii_label_files, None, None, quantile_normalization=True
            )


if __name__ == '__main__':
    import torchreg
    import numpy as np
    ds = HippocampusMRDataset("../HippocampusMR", 'test', True)
    
    print('train', len(HippocampusMRDataset("../HippocampusMR", 'train', True)))
    print('val', len(HippocampusMRDataset("../HippocampusMR", 'val', True)))
    print('test', len(HippocampusMRDataset("../HippocampusMR", 'test', True)))
    dice_overlap = torchreg.metrics.DiceOverlap(
            classes=[0,1,2])
    dos = []
    random.seed(42)
    for i in range(100):
        idx = random.randint(0,len(ds))
        (i1, s1), (i2, s2) = ds[idx]
        do = dice_overlap(s1, s2)
        dos.append(do)
    print('mean dice overlap:', np.mean(dos))
