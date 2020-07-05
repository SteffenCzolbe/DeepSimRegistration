import torch
import torch.nn as nn
import torch.nn.functional as F
import torchreg
import numpy as np
import math


class NCC(nn.Module):
    """
    Local (over window) normalized cross correlation loss. Normalized to window [0,2], with 0 being perfect match.
    """

    def __init__(self, window=9, squared=False, eps=1e-6):
        super().__init__()
        self.win = window
        self.squared = squared
        self.eps = eps

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
        if self.squared:
            cc = cross**2 / (var0 * var1).clamp(self.eps)
        else:
            cc = cross / (var0.clamp(self.eps)**0.5 * var1.clamp(self.eps)**0.5)

        # mean and invert for minimization
        return -torch.mean(cc) + 1

class DeepSim(nn.Module):
    """
    Deep similarity metric
    """
    def __init__(self, seg_model, eps=1e-6):
        super().__init__()
        self.seg_model = seg_model
        self.eps = eps

        # fix params
        for param in self.seg_model.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        # set to eval (deactivate dropout)
        self.seg_model.eval()

        # extract features
        feats0 = self.seg_model.extract_features(y_true)
        feats1 = self.seg_model.extract_features(y_pred)
        losses = []
        for feat0, feat1 in zip(feats0, feats1):
            # calculate cosine similarity
            prod_ab = torch.sum(feat0 * feat1, dim=1)
            norm_a = torch.sum(feat0**2, dim=1).clamp(self.eps)**0.5
            norm_b = torch.sum(feat1**2, dim=1).clamp(self.eps)**0.5
            cos_sim = prod_ab / (norm_a * norm_b)
            losses.append(torch.mean(cos_sim))

        # mean and invert for minimization
        return -torch.stack(losses).mean() + 1

if __name__ == '__main__':
    import numpy as np
    import torchreg
    torchreg.settings.set_ndims(2)
    npz = np.load('invalid_loss_val.npz')
    I_m = torch.tensor(npz['I_m'])
    I_1 = torch.tensor(npz['I_1'])
    ncc = NCC(window=9)
    import ipdb
    ipdb.set_trace()
    ncc(I_m, I_1)
