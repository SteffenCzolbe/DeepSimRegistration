"""
Metrics For annotated Images
"""
import torch
import torch.nn as nn
from .metrics import DiceOverlap

class SegmentationDiceOverlap(nn.Module):
    """
    calculates the mean dice overlap of the given classes

    WARNING:
    return of forward function is a tensor containing the dice overlap per image pair.
    If an image does not have a segmentation mask, -1 is returned at it's position instead.

    Parameters:
        classes: list of classes to consider, e.g. [0, 1]
    """
    def __init__(self, classes):
        super().__init__()
        self.dice_metric = DiceOverlap(classes=classes)

    def forward(self, annoated_image0, annoated_image1):
        dice_overlaps = []

        for i in range(len(annoated_image0)):
            # calculate segmentation dice overlap, if a segmentation mask is available
            if annoated_image0.segmentation[i] is not None and annoated_image1.segmentation[i] is not None:
                dice_overlap = self.dice_metric(
                    annoated_image0.segmentation[i], annoated_image1.segmentation[i]
                )
                dice_overlaps.append(dice_overlap)
            else:
                dice_overlaps.append(torch.tensor(-1).type_as(annoated_image0.intensity))
        return torch.stack(dice_overlaps)

class TargetRegistrationError(nn.Module):
    """
    calculates the mean dice overlap of the given classes

    WARNING:
    return of forward function is a tensor containing the dice overlap per image pair.
    If an image does not have a segmentation mask, -1 is returned at it's position instead.

    Parameters:
        norm: 'l1' or 'l2'
    """
    def __init__(self, norm='l1'):
        super().__init__()
        self.norm = norm

    def forward(self, annoated_image0, annoated_image1):
        target_registration_errors = []

        for i in range(len(annoated_image0)):
            if annoated_image0.landmarks[i] is not None and annoated_image1.landmarks[i] is not None:
                # calculate euclidean landmark error
                euclidean_dist = (
                    torch.mean(
                        torch.sum(
                            (annoated_image0.landmarks[i] - annoated_image1.landmarks[i]) ** 2, dim=0
                        )
                        ** 0.5
                    )
                )

                if self.norm == 'l2':
                    euclidean_dist = euclidean_dist**2

                target_registration_errors.append(euclidean_dist)
            else:
                target_registration_errors.append(torch.tensor(-1).type_as(annoated_image0.intensity))

        return torch.stack(target_registration_errors)        
