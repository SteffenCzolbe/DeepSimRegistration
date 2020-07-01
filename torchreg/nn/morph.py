"""
Morphing of AnnotatedImages
"""
import torch.nn as nn
from .layers import SpatialTransformer, GridSampler
from torchreg.types import AnnotatedImage, TensorList
from torchreg.settings import settings



class ImageTransform(nn.Module):
    """
    Applies a transformation to an batch of Annotated Images
    """
    def __init__(self):
        """
        Transforms the segmentation mask
        Same as spatial stransformer, but uses nearest interpolation and rounds to integers
        """
        super().__init__()
        self.image_transform = SpatialTransformer()
        self.segmentation_transform = SegmentationTransform()
        self.landmark_transform = LandmarkTransform()

    def forward(self, source, flow, inv_flow, target=None):
        """
        Parameters:
        source: image to morph
        flow: forward-flow (for image and segmentation mask)
        inv_flow: inverse flow (for landmarks)
        target: target to morph to. used to extract context information. optional
        """
        intensity = self.image_transform(source.intensity, flow)
        mask = self.image_transform(source.mask, flow)
        segmentation = self.segmentation_transform(source.segmentation, flow)
        landmarks = self.landmark_transform(source.landmarks, inv_flow)

        if target is None:
            contex = source.context
        else:
            contex = [{"moving": ctx_mov, "fixed": ctx_fix} for ctx_mov, ctx_fix in zip(source.context, target.context)]

        return AnnotatedImage(intensity, mask, segmentation, landmarks, context=contex)




class SegmentationTransform(nn.Module):
    """
    A layer that transforms an optional segmentation mask
    """

    def __init__(self):
        """
        Transforms the segmentation mask
        Same as spatial stransformer, butuses nearest interpolation and rounds to integers
        """
        super().__init__()
        self.segmentation_transform = SpatialTransformer(mode="nearest")

    def forward(self, segmentation, flow):
        """
        Warps the segmentation masks along the flow.
        
        As not every sample has a segmentation mask, an iterative approach is chosen.
        
        Parameters:
            segmentation: The segmentation masks, where the first dimension is the batch. 
            The batch-dimension needs to be iterable. Samples of the batch of size 0 are ignored. 
        """
        segmentations_transformed = []
        # iterate over batch
        for i in range(len(segmentation)):
            if segmentation[i] is not None:
                # morph the segmentation mask
                segmentation_transformed = self.segmentation_transform(
                    segmentation[i].unsqueeze(0), flow[[i]]
                )
                # discretize segmentation classes to account for numerical inaccuracy
                segmentation_transformed = segmentation_transformed.round()
                # save result
                segmentations_transformed.append(segmentation_transformed[0])
            else:
                # copy empty segmentation mask
                segmentations_transformed.append(segmentation[i])
        return TensorList.from_list(segmentations_transformed)


class LandmarkTransform(nn.Module):
    """
    A layer that transforms optional landmark annotations
    """

    def __init__(self):
        """
        Transforms landmarks.
        """
        super().__init__()
        self.sampler = GridSampler(mode="bilinear")
        self.ndims = settings.get_ndims()

    def forward(self, landmarks, flow):
        """
        Warps the segmentation masks along the flow.
        
        As not every sample has a segmentation mask, an iterative approach is chosen.
        
        Parameters:
            landmarks: The landmark annotations, where the first dimension is the batch, the 2nd the channel and the 3rd the length (count of landmarks). 
            The batch-dimension needs to be iterable. Samples of the batch of size 0 are ignored. 
        """
        landmarks_transformed = []
        # iterate over batch
        for i in range(len(landmarks)):
            if landmarks[i] is not None:
                # reform from C x N to 1 x C x N x 1 x 1
                C, N = landmarks[i].shape
                if self.ndims == 2:
                    landmark = landmarks[i].view(1, C, N, 1)
                elif self.ndims == 3:
                    landmark = landmarks[i].view(1, C, N, 1, 1)
                # morph the landmarks
                landmark_displacement = self.sampler(flow[[i]], landmark)
                landmark_transformed = landmark + landmark_displacement
                # unpack back to C x N and add to list
                landmarks_transformed.append(landmark_transformed.squeeze())
            else:
                # copy empty landmark list
                landmarks_transformed.append(landmarks[i])
        return TensorList.from_list(landmarks_transformed)
