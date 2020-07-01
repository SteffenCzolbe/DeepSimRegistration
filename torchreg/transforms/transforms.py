"""
Transforms to work on tuples of annotated images. 
Multiple transforms can be combined with torchvision.transforms.Compose
"""

import torch
import numpy as np
from .functional import *
from torchvision.transforms import Compose
import torch.nn as nn
import torch.nn.functional as F
import torchreg.nn as tnn
import torchreg.settings as settings


class ToTensor:
    """
    transforms an image tuple to tensors
    """

    def __call__(self, image_tuple):
        for image in image_tuple:
            self.transform_annotated_image(image)
        return image_tuple

    def transform_annotated_image(self, annotated_image):
        annotated_image.intensity = volumetric_image_to_tensor(
            annotated_image.intensity
        )
        annotated_image.mask = volumetric_image_to_tensor(annotated_image.mask)
        if annotated_image.segmentation is not None:
            annotated_image.segmentation = volumetric_image_to_tensor(
                annotated_image.segmentation
            )
        if annotated_image.landmarks is not None:
            annotated_image.landmarks = landmarks_to_tensor(annotated_image.landmarks)
        return annotated_image


class ToNumpy:
    """
    transforms an image tuple to numpy arrays
    """

    def __call__(self, image_tuple):
        for image in image_tuple:
            self.transform_annotated_image(image)
        return image_tuple

    def transform_annotated_image(self, annotated_image):
        annotated_image.intensity = image_to_numpy(annotated_image.intensity)
        annotated_image.mask = image_to_numpy(annotated_image.mask)
        if annotated_image.segmentation is not None:
            annotated_image.segmentation = image_to_numpy(annotated_image.segmentation)
        if annotated_image.landmarks is not None:
            annotated_image.landmarks = landmarks_to_numpy(annotated_image.landmarks)
        return annotated_image


class RandomDiffeomorphic(nn.Module):
    """
    applies a random diffeomorphic transformation to the Tuple of AnnotatedImages
    This transform is implemented as a nn.Module, and can also be used within the network architecture to be executed on GPU.
    """

    def __init__(self, p=0.5, m=2.0, r=8):
        """
        Parameters:
            p: probability to apply a transform to each image
            m: stddev of the transformations (pixels)
            r: resolution fectos on which to generate the random transformation vectors on
        """
        super().__init__()
        self.p = p
        self.m = m
        self.r = r
        self.integrate = tnn.FlowIntegration(nsteps=5)
        self.transform = tnn.ImageTransform()

    def forward(self, image_tuple):
        for image in image_tuple:
            self.transform_annotated_image(image)
        return image_tuple

    def transform_annotated_image(self, annotated_image):
        if np.random.rand() > self.p:
            return annotated_image

        # dimensionality stuff
        B = annotated_image.intensity.shape[0]
        C = settings.get_ndims()
        size = annotated_image.intensity.shape[2:]
        # keep B and C, scale down H,W,D,...
        size_small = [B, C] + [s // self.r for s in size]

        # create some noise at lower resolution
        flow = torch.randn(size_small).type_as(annotated_image.intensity) * self.m
        # upsample to full resolution
        mode = tnn.interpol_mode("linear")
        flow = F.interpolate(flow, size=size, mode=mode, align_corners=False)
        # integrate for smoothness
        pos_flow = self.integrate(flow)
        neg_flow = self.integrate(-flow)
        # apply to the image

        return self.transform(annotated_image, pos_flow, neg_flow)

