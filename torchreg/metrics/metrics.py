import torch
import torch.nn as nn
import torch.nn.functional as F
import torchreg
import numpy as np
import math


class NCC(nn.Module):
    """
    Local (over window) normalized cross correlation loss. Normalized to window [0,1], with 0 being perfect match.

    We follow the NCC definition from the paper "VoxelMorph: A Learning Framework for Deformable Medical Image Registration",
    which implements it via the coefficient of determination (R2 score). 
    This is strictly the squared normalized cross-correlation, or squared cosine similarity.

    NCC over two image pacthes I, J of size N is calculated as
    NCC(I, J) = 1/N * [sum_n=1^N (I_n - mean(I)) * (J_n - mean(J))]^2 / [var(I) * var(J)]

    The output is rescaled to the interval [0..1], best match at 0.

    """

    def __init__(self, window=5):
        super().__init__()
        self.win = window

    def forward(self, y_true, y_pred):
        def compute_local_sums(I, J):
            # calculate squared images
            I2 = I * I
            J2 = J * J
            IJ = I * J

            # take sums
            I_sum = conv_fn(I, filt, stride=stride, padding=padding)
            J_sum = conv_fn(J, filt, stride=stride, padding=padding)
            I2_sum = conv_fn(I2, filt, stride=stride, padding=padding)
            J2_sum = conv_fn(J2, filt, stride=stride, padding=padding)
            IJ_sum = conv_fn(IJ, filt, stride=stride, padding=padding)

            # take means
            win_size = np.prod(filt.shape)
            u_I = I_sum / win_size
            u_J = J_sum / win_size

            # calculate cross corr components
            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

            return I_var, J_var, cross

        # get dimension of volume
        ndims = torchreg.settings.get_ndims()
        channels = y_true.shape[1]

        # set filter
        filt = torch.ones(channels, channels, *([self.win] * ndims)).type_as(y_true)

        # get convolution function
        conv_fn = getattr(F, "conv%dd" % ndims)
        stride = 1
        padding = self.win // 2

        # calculate cc
        var0, var1, cross = compute_local_sums(y_true, y_pred)
        cc = cross * cross / (var0 * var1 + 1e-5)

        # mean and invert for minimization
        return - torch.mean(cc)


class MaskedNCC(nn.Module):
    """
    Masked Normalized Cross-coralation. 
    """

    def __init__(self, window=5):
        super().__init__()
        self.ncc = NCC(window=window)

    def forward(self, y_true, y_pred, mask0, mask1):
        # we first null out any areas outside of the masks, s o that the boarders are the same
        y_true = y_true * mask0 * mask1
        y_pred = y_pred * mask0 * mask1
        # perform ncc
        return self.ncc(y_true, y_pred)


class MSE(nn.Module):
    """
    Mean squared error loss.
    """

    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class MaskedMSE(nn.Module):
    """
    Masked Mean squared error. 
    """

    def __init__(self):
        super(MaskedMSE, self).__init__()

    def forward(self, y_true, y_pred, mask0, mask1):
        sq_error = (y_true - y_pred) ** 2
        masked_sq_error = sq_error * mask0 * mask1
        return torch.mean(masked_sq_error)


class GradNorm(nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty="l2", reduction="mean"):
        super(GradNorm, self).__init__()
        self.penalty = penalty
        self.ndims = torchreg.settings.get_ndims()
        self.reduction = reduction

    def forward(self, flow):
        # pad flow
        flow = F.pad(flow, [0, 1] * self.ndims, mode="replicate")
        # get finite differences
        if self.ndims == 2:
            dx = torch.abs(flow[:, :, 1:, :-1] - flow[:, :, :-1, :-1])
            dy = torch.abs(flow[:, :, :-1, 1:] - flow[:, :, :-1, :-1])
        elif self.ndims == 3:
            dx = torch.abs(flow[:, :, 1:, :-1, :-1] - flow[:, :, :-1, :-1, :-1])
            dy = torch.abs(flow[:, :, :-1, 1:, :-1] - flow[:, :, :-1, :-1, :-1])
            dz = torch.abs(flow[:, :, :-1, :-1, 1:] - flow[:, :, :-1, :-1, :-1])

        # square
        if self.penalty == "l2":
            dx = dx ** 2
            dy = dy ** 2
            if self.ndims == 3:
                dz = dz ** 2

        d = dx + dy + (dz if self.ndims == 3 else 0)
        d /= self.ndims
        if self.reduction == "none":
            # mean over channels. Keep spatial dimensions
            return torch.mean(d, dim=1, keepdim=True)
        elif self.reduction == "mean":
            return torch.mean(d)
        elif self.reduction == "sum":
            return torch.sum(d)


class PixelArea(nn.Module):
    def __init__(self, reduction="mean"):
        super(PixelArea, self).__init__()
        self.idty = torchreg.nn.Identity()
        self.reduction = reduction
        if torchreg.settings.get_ndims() != 2:
            raise Exception("Only 2D supported by this operation.")

    def forward(self, flow):
        """
        calculates the area of each pixel after the flow is applied
        """

        def determinant_2d(x, y):
            return x[:, [0]] * y[:, [1]] - x[:, [1]] * y[:, [0]]

        # pad flow
        flow = F.pad(flow, (1, 1, 1, 1), mode="replicate")

        # map to target domain
        transform = flow + self.idty(flow)

        # calculate area of upper left triangle of each grid cell
        dx = transform[:, :, 2:, 1:-1, ...] - transform[:, :, 1:-1, 1:-1, ...]
        dy = transform[:, :, 1:-1, 2:, ...] - transform[:, :, 1:-1, 1:-1, ...]
        area_upper_triag = 0.5 * determinant_2d(dx, dy).abs()

        # calculate area of lower right triangle of each grid cell
        dx = transform[:, :, :-2, 1:-1, ...] - transform[:, :, 1:-1, 1:-1, ...]
        dy = transform[:, :, 1:-1, :-2, ...] - transform[:, :, 1:-1, 1:-1, ...]
        area_lower_triag = 0.5 * determinant_2d(dx, dy).abs()

        area = area_upper_triag + area_lower_triag
        if self.reduction == "none":
            return area
        elif self.reduction == "mean":
            return torch.mean(area)
        elif self.reduction == "sum":
            return torch.sum(area)


class DiceOverlap(nn.Module):
    def __init__(self, classes):
        """
        calculates the mean dice overlap of the given classes
        This loss metric is not suitable for training. Use the SoftdiceOverlap for training instead.
        Parameters:
            classes: list of classes to consider, e.g. [0, 1]
        """
        super(DiceOverlap, self).__init__()
        self.classes = classes

    def cast_to_int(self, t):
        if t.dtype == torch.float:
            return t.round().int()
        else:
            return t

    def forward(self, y_true, y_pred):
        y_true = self.cast_to_int(y_true)
        y_pred = self.cast_to_int(y_pred)
        dice_overlaps = []
        for label in self.classes:
            mask0 = y_true == label
            mask1 = y_pred == label
            intersection = (mask0 * mask1).sum()
            union = mask0.sum() + mask1.sum()
            dice_overlap = 2.0 * intersection / (union + 1e-6)
            dice_overlaps.append(dice_overlap)
        return torch.stack(dice_overlaps).mean()


class SoftDiceOverlap(nn.Module):
    def __init__(self):
        """
        calculates the mean soft dice overlap of one-hot encoded feature maps.
        This loss metric is suitable for training
        Parameters:
            classes: list of classes to consider, e.g. [0, 1]
        """
        super(SoftDiceOverlap, self).__init__()

    def forward(self, y_true, y_pred):
        # calculate union
        union = y_true * y_pred

        # sum over B, D, H, W
        sum_dims = [0, 2, 3] if torchreg.settings.get_ndims() == 2 else [0, 2, 3, 4]
        s_union = torch.sum(union, dim=sum_dims)
        s_y_true = torch.sum(y_true, dim=sum_dims)
        s_y_pred = torch.sum(y_pred, dim=sum_dims)

        # calculate dice per class, mean over classes
        dice = 2 * s_union / (s_y_true + s_y_pred + 1e-6)
        return torch.mean(dice)
