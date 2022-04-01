import os
import pandas as pd
from .nii_dataset import NiiDataset, NiiAtlasDataset
import torchreg.transforms.functional as f


def BrainMRIDataset(path: str, split: str, pairs=False):
    """
    creates the BrainMRI dataset.

    Parameters:
        path: path to the dataset folder
        split: the dataset split. eg: 'train'
        pairs: bool, if True return pairs for registration
    """

    # load config
    df = pd.read_csv(os.path.join(path, "metadata.csv"), dtype=str)
    df.set_index("subject_id", inplace=True)
    subjects = list(df[df["SPLIT"] == split].index)
    image_nii_files = list(
        map(lambda s: os.path.join(path, "data", s, "brain_aligned.nii.gz"), subjects)
    )
    image_nii_label_files = list(
        map(
            lambda s: os.path.join(path, "data", s, "seg_coalesced_aligned.nii.gz"),
            subjects,
        )
    )
    atlas_nii_file = os.path.join(path, "atlas", "brain_aligned.nii.gz")
    atlas_nii_label_file = os.path.join(path, "atlas", "seg_coalesced_aligned.nii.gz")

    min_intensity = 0
    max_intensity = 128

    if pairs:
        return NiiAtlasDataset(
            atlas_nii_file,
            atlas_nii_label_file,
            image_nii_files,
            image_nii_label_files,
            min_intensity,
            max_intensity,
        )
    else:
        return NiiDataset(
            image_nii_files, image_nii_label_files, min_intensity, max_intensity
        )

