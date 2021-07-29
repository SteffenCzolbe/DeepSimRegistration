import numpy as np
import ants
import torchreg
import torch
import torch.nn.functional as F
import argparse
import os
from src.registration_model import RegistrationModel
from src.segmentation_model import SegmentationModel
from src.autoencoder_model import AutoEncoderModel
from typing import *


def load_model(dataset, loss_function):
    path = os.path.join("./weights/", dataset, "registration", loss_function)
    checkpoint_path = os.path.join(path, "weights.ckpt")
    return RegistrationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)


def get_dataloader(dataset):
    # load a model to steal the dataloader
    model = load_model(dataset, "l2")
    class_cnt = model.dataset_config("classes")
    return model.test_dataloader(), class_cnt


def get_img_pair(dataloader, index, feature_extractor=None):
    # load image pair
    dataset = dataloader.dataset
    (I_0, S_0), (I_1, S_1) = dataset[index]

    # augment with deep features
    if feature_extractor:
        I_0 = augment_tensor_with_deep_features(I_0, feature_extractor)
        I_1 = augment_tensor_with_deep_features(I_1, feature_extractor)

    return (I_0, S_0), (I_1, S_1)


def torch_img_to_ants(torch_tensor: torch.Tensor) -> List[ants.core.ANTsImage]:
    """Transform a torch tensor of a single image (no batch dimension) to a list of ANTS Images, one image per channe;

    Args:
        torch_tensor (torch.Tensor): [description]

    Returns:
        List[ants.core.ANTsImage]: [description]
    """
    # to numpy
    numpy_array = torchreg.transforms.functional.image_to_numpy(torch_tensor)
    if numpy_array.dtype == np.int64:
        numpy_array = numpy_array.astype(np.uint8)

    dims = torchreg.settings.get_ndims()
    has_channels = len(numpy_array.shape) > dims
    if has_channels:
        C = numpy_array.shape[-1]
        return [ants.from_numpy(numpy_array[..., c].copy()) for c in range(C)]
    else:
        return [ants.from_numpy(numpy_array)]


def ants_img_to_torch(ants_img: ants.core.ANTsImage) -> torch.Tensor:
    """Transforms an ANTS image into a torch tensor (no batch-dimension)

    Args:
        ants_img (ants.core.ANTsImage): [description]

    Returns:
        torch.Tensor: [description]
    """
    numpy_array = ants_img.numpy(single_components=True)
    from_dtype = numpy_array.dtype
    if from_dtype == np.float32:
        to_dtype = torch.float32
    elif from_dtype == np.uint8:
        to_dtype = torch.int64
    torch_tensor = torchreg.transforms.functional.volumetric_image_to_tensor(
        numpy_array, dtype=to_dtype)
    return torch_tensor


def load_feature_extractor(device, path: str, feature_extractor_model='seg'):
    if feature_extractor_model == "seg":
        feature_extractor = SegmentationModel.load_from_checkpoint(
            path
        )
    elif feature_extractor_model == "ae":
        feature_extractor = AutoEncoderModel.load_from_checkpoint(
            path
        )
    elif feature_extractor_model == "none":
        return None
    else:
        raise Exception(
            f"unknown feature_extractor_model: '{feature_extractor_model}'")
    feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor


def augment_tensor_with_deep_features(img: torch.Tensor, feature_extractor) -> torch.Tensor:
    device = feature_extractor.device
    interpol_model = 'bicubic' if torchreg.settings.get_ndims() == 2 else 'trilinear'
    spatial_dims = [2, 3] if torchreg.settings.get_ndims() == 2 else [2, 3, 4]
    img = img.to(device)

    # add batch
    img = img.unsqueeze(0)
    with torch.no_grad():
        # extract features
        feats = feature_extractor.extract_features(img)

        # upscale all layers to full resolution
        feats = [F.interpolate(feat, size=img.shape[2:], mode=interpol_model, align_corners=False)
                 for feat in feats]

        # stack along channel dimension
        feats = torch.cat(feats, dim=1)

        # add raw image to features
        feats = torch.cat([img, feats], dim=1)

        # normalize by channel
        channel_norms = (feats**2).sum(dim=spatial_dims, keepdim=True) ** 0.5
        feats = feats / channel_norms

    # remove batch
    return feats.squeeze(0)


def register_and_eval_multichannel_tensors(moving: torch.Tensor, moving_seg: torch.Tensor, fixed: torch.Tensor, fixed_seg: torch.Tensor, class_cnt: int) -> float:
    # map tensors to ants
    moving_ants = torch_img_to_ants(moving)
    fixed_ants = torch_img_to_ants(fixed)
    moving_seg_ants = torch_img_to_ants(moving_seg)[0]
    fixed_seg_ants = torch_img_to_ants(fixed_seg)[0]

    # set up metric for additional channels
    channels = len(moving_ants)
    if channels == 1:
        multivariate_extras = None
    else:
        print(f'registering {channels} channels')
        multivariate_extras = [
            ('meansquares', fixed_ants[c], moving_ants[c], 1, None) for c in range(1, channels)]

    # register with SyN
    # docs: https://antspy.readthedocs.io/en/latest/registration.html
    ret = ants.registration(fixed=fixed_ants[0],
                            moving=moving_ants[0],
                            type_of_transform='SyNOnly',
                            multivariate_extras=multivariate_extras,
                            flow_sigma=3,
                            total_sigma=0,
                            reg_iterations=(80, 0, 0),  # TODO
                            verbose=True)

    # warp segmentation
    # fixed image required to define the domain
    morphed_seg_ants = ants.apply_transforms(fixed_seg_ants, moving_seg_ants, ret["fwdtransforms"],
                                             interpolator='nearestNeighbor', imagetype=0, verbose=False)

    # map registered image back to tensor
    morphed_seg = ants_img_to_torch(morphed_seg_ants)

    # calculate Dice overlap
    dice_overlap = torchreg.metrics.DiceOverlap(
        classes=list(range(class_cnt))
    )

    print("Dice overlap no registration:")
    print(dice_overlap(moving_seg.unsqueeze(0), fixed_seg.unsqueeze(0)))
    print("Dice overlap with registration:")
    print(dice_overlap(morphed_seg.unsqueeze(0), fixed_seg.unsqueeze(0)))

    return dice_overlap(morphed_seg.unsqueeze(0), fixed_seg.unsqueeze(0))


def main(hparams):
    # setup
    dataloader, class_cnt = get_dataloader(hparams.dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_extractor = load_feature_extractor(
        device, hparams.feature_extractor_weights, feature_extractor_model=hparams.feature_extractor)

    # load images
    (I_0, S_0), (I_1, S_1) = get_img_pair(
        dataloader, 0, feature_extractor)

    dice_overlap = register_and_eval_multichannel_tensors(
        I_0, S_0, I_1, S_1, class_cnt)


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices=['brain-mri', 'platelet-em', 'phc-u373'], help="dataset"
    )
    parser.add_argument(
        "--feature_extractor", type=str, choices=['none', 'seg', 'ae'], default='none', help="feature_extractor"
    )
    parser.add_argument(
        "--feature_extractor_weights", type=str, help="path to feature extractor weights"
    )
    hparams = parser.parse_args()
    main(hparams)
