import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchreg
from torchvision import models


class DeepSim(nn.Module):
    """
    Deep similarity metric
    """

    def __init__(self, seg_model, eps=1e-6, levels='all', reduction='mean', zero_mean=False):
        super().__init__()
        self.seg_model = seg_model
        self.eps = eps
        self.levels = levels
        self.reduction = reduction
        self.zero_mean = zero_mean

        # fix params
        for param in self.seg_model.parameters():
            param.requires_grad = False

    
    def _calculate_cos_sim(self, feat0, feat1):
        prod_ab = torch.sum(feat0 * feat1, dim=1)
        norm_a = torch.sum(feat0 ** 2, dim=1).clamp(self.eps) ** 0.5
        norm_b = torch.sum(feat1 ** 2, dim=1).clamp(self.eps) ** 0.5
        cos_sim = prod_ab / (norm_a * norm_b)
        return cos_sim

    def _zero_mean_feat(self, feats):
        zero_mean_feats = []
        for feat in feats:
            feat_mean = torch.mean(feat,1) 
            feat_mean = feat_mean.unsqueeze(1).expand_as(feat)
            zero_mean_feat = feat - feat_mean
            zero_mean_feats.append(zero_mean_feat)
        return zero_mean_feats

    def forward(self, y_true, y_pred):
        # set to eval (deactivate dropout)
        self.seg_model.eval()

        # extract features
        feats0 = self.seg_model.extract_features(y_true)
        feats1 = self.seg_model.extract_features(y_pred)
        losses = []

        if self.zero_mean:
            feats0 = self._zero_mean_feat(feats0)
            feats1 = self._zero_mean_feat(feats1)

        for i, (feat0, feat1) in enumerate(zip(feats0, feats1)):
            # calculate cosine similarity
            if self.levels == 'all':
                cos_sim = self._calculate_cos_sim(feat0, feat1)
                if self.reduction == 'mean':
                    losses.append(torch.mean(cos_sim))
                else:
                    losses.append(cos_sim)
            else:
                if i in self.levels:
                    cos_sim = self._calculate_cos_sim(feat0, feat1)
                    if self.reduction == 'mean':
                        losses.append(torch.mean(cos_sim))
                    else:
                        losses.append(cos_sim)

        if self.reduction == 'mean':
            return -torch.stack(losses).mean() + 1
        else:
            return losses

class DeepSim_v2(nn.Module):
    """
    Deep similarity metric
    Alternative implementation
    """

    def __init__(self, feature_extractor_model, eps=1e-6):
        super().__init__()
        # feature extractor model variable called seg_model for backwards compatibility. Can be any model implementing the extract_features method
        self.seg_model = feature_extractor_model
        self.eps = eps
        self.transformer = torchreg.nn.SpatialTransformer()

        # fix params of feature extractor
        for param in self.seg_model.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        """DeepSim Loss. Taking the target and a morphed image. Extract features from the morphed image to calculate similarity.

        Args:
            y_true (torch.Tensor): The true target image. 
            y_pred (torch.Tensor): The predicted (transformed) image

        Returns:
            torch.Tensor: the loss
        """
        # set to eval (deactivate dropout)
        self.seg_model.eval()

        # extract features
        feats0 = self.seg_model.extract_features(y_true)
        feats1 = self.seg_model.extract_features(y_pred)
        return self._compare_feat_maps(feats0, feats1)

    def first_warp_then_extract_features(self, I_0, I_1, flow):
        """Alternative interface to self.forward().

        Args:
            I_0 (torch.Tensor): The moving image
            I_1 (torch.Tensor): The fixed image
            flow (torch.Tensor): the transformation

        Returns:
            torch.Tensor: the loss
        """
        I_m = self.transformer(I_0, flow)
        return self.forward(I_1, I_m)

    def first_extract_features_then_warp(self, I_0, I_1, flow):
        """ Extracts features before warping.

        Args:
            I_0 (torch.Tensor): The moving image
            I_1 (torch.Tensor): The fixed image
            flow (torch.Tensor): the transformation

        Returns:
            torch.Tensor: the loss
        """
        # set to eval (deactivate dropout)
        self.seg_model.eval()

        # extract features
        feats0 = self.seg_model.extract_features(I_0)
        feats1 = self.seg_model.extract_features(I_1)

        def apply_downscaled_transform(image, flow):
            img_size = image.shape[2]
            flow_size = flow.shape[2]
            scale_factor = img_size / flow_size
            scaled_flow = torch.nn.functional.interpolate(flow,
                                                          scale_factor=scale_factor, mode='nearest')
            scaled_flow *= scale_factor
            return self.transformer(image, scaled_flow)

        # warp features
        feats0 = [apply_downscaled_transform(feat, flow) for feat in feats0]

        # calculate loss
        return self._compare_feat_maps(feats0, feats1)

    def _compare_feat_maps(self, feats0, feats1):
        losses = [self._cosine_sim(feat0, feat1).mean()
                  for feat0, feat1 in zip(feats0, feats1)]

        # mean and invert for minimization. Add +1 so loss >=0
        return -torch.stack(losses).mean() + 1

    def _cosine_sim(self, a, b):
        prod_ab = torch.sum(a * b, dim=1)
        norm_a = torch.sum(a ** 2, dim=1).clamp(self.eps) ** 0.5
        norm_b = torch.sum(b ** 2, dim=1).clamp(self.eps) ** 0.5
        cos_sim = prod_ab / (norm_a * norm_b)
        return cos_sim

class NCC(nn.Module):
    """
    Local (over window) normalized cross correlation loss. Normalized to window [0,2], with 0 being perfect match.
    """

    def __init__(self, window=9, squared=False, eps=1e-6, reduction='mean'):
        super().__init__()
        self.win = window
        self.squared = squared
        self.eps = eps
        self.reduction = reduction

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
            cc = cross ** 2 / (var0 * var1).clamp(self.eps)
        else:
            cc = cross / (var0.clamp(self.eps) ** 0.5 * var1.clamp(self.eps) ** 0.5)

        # mean and invert for minimization
        if self.reduction == 'mean':
            return -torch.mean(cc) + 1
        else:
            return cc

class VGGFeatureExtractor(nn.Module):
    """
    pretrained VGG-net as a feature extractor
    """

    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.N_slices = 3
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # pad x to RGB input
        x = torch.cat([x, x, x], dim=1)
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3]

    def extract_features(self, x):
        return self(x)
    
    
class MIND_loss(torch.nn.Module):
    def __init__(self, radius:int=2, dilation:int=2):
        """
        Implementation of the MIND-SSC loss function
        See http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for the MIND-SSC descriptor. This ignores the center voxel, but compares adjacent voxels of the 6-neighborhood with each other.
        See http://mpheinrich.de/pub/MEDIA_mycopy.pdf for the original MIND loss function.

        Implementation retrieved from 
        https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/TransMorph/losses.py
        Annotation/Comments and support for 2d by Steffen Czolbe.

        Parameters:
            radius (int): determines size of patches around members of the 6-neighborhood
            dilation (int): determines spacing of members of the 6-neighborhood from the center voxel
        """
        super(MIND_loss, self).__init__()
        self.radius = radius
        self.dilation = dilation

    def pdist_squared(self, x):
        # for a list of length N of interger-valued pixel-coordinates, return an NxN matrix containing the squred euclidian distance between them:
        # 0: coordinates are the same
        # 1: coordinates are neighbours
        # 2: coordinates are diagonal
        # 4: coordinates are opposide, 2 apart

        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC3d(self, img):
        B, C, H, W, D = img.shape
        assert H>1 and W>1 and D>1, "Use 2d implementation for 2d data"

        # Radius: determines size of patches around members of the 6-neighborhood, square kernel
        kernel_size = self.radius * 2 + 1

        # define neighborhood for the self-similarity pattern. These coordinates are centered on [1, 1, 1]
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances between neighborhood coordinates
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        # we compare adjacent neighborhood pixels (squared distance ==2), and exploid siymmetry to only calculate each pair once (x>y).
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernels to efficiently implement the pairwhise-patch based differences
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :] # 12x3 matrix. Pairing-pixel-coordinates of the first image
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :] # 12x3 matrix. Pairing-pixel-coordinates of the second image
        mshift1 = torch.zeros(12, 1, 3, 3, 3).to(img.device) # shifting-kernels, one channel to 12 channels. Each 3x3x3 kernel has ony a single 1
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).to(img.device)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = nn.ReplicationPad3d(self.dilation) # Padding to account for borders.
        rpad2 = nn.ReplicationPad3d(self.radius)

        # compute patch-ssd

        # shift-align all 12 patch pairings, implemented via convolution
        h1 = F.conv3d(rpad1(img), mshift1, dilation=self.dilation)
        h2 = F.conv3d(rpad1(img), mshift2, dilation=self.dilation)
        # calculate difference 
        diff = rpad2((h1 - h2) ** 2)
        # convolve difference patches via averaging. This makes the loss magnitude invriant of patch size.
        ssd = F.avg_pool3d(diff, kernel_size, stride=1)


        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0] # normalize by substracting lowest value
        mind_var = torch.mean(mind, 1, keepdim=True) # Mean across neighborhood pixel pairings (channels)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item()) # remove outliers
        mind /= mind_var
        mind = torch.exp(-mind)

        return mind


    def MINDSSC2d(self, img):
        # Radius: determines size of patches around members of the 4-neighborhood, square kernel
        kernel_size = self.radius * 2 + 1

        # define neighborhood for the self-similarity pattern. These coordinates are centered on [1, 1]
        four_neighbourhood = torch.Tensor([[0, 1],
                                            [1, 0],
                                            [2, 1],
                                            [1, 2]]).long()

        # squared distances between neighborhood coordinates
        dist = self.pdist_squared(four_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        # we compare adjacent neighborhood pixels (squared distance ==2), and exploid siymmetry to only calculate each pair once (x>y).
        x, y = torch.meshgrid(torch.arange(4), torch.arange(4))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernels to efficiently implement the pairwhise-patch based differences
        idx_shift1 = four_neighbourhood.unsqueeze(1).repeat(1, 4, 1).view(-1, 2)[mask, :] # 4x2 matrix. Pairing-pixel-coordinates of the first image
        idx_shift2 = four_neighbourhood.unsqueeze(0).repeat(4, 1, 1).view(-1, 2)[mask, :] # 4x2 matrix. Pairing-pixel-coordinates of the second image
        mshift1 = torch.zeros(4, 1, 3, 3).to(img.device) # shifting-kernels, one channel to 4 channels. Each 3x3 kernel has ony a single 1
        mshift1.view(-1)[torch.arange(4) * 9 + idx_shift1[:, 0] * 3 + idx_shift1[:, 1]] = 1
        mshift2 = torch.zeros(4, 1, 3, 3).to(img.device)
        mshift2.view(-1)[torch.arange(4) * 9 + idx_shift2[:, 0] * 3 + idx_shift2[:, 1]] = 1
        rpad1 = nn.ReplicationPad2d(self.dilation) # Padding to account for borders.
        rpad2 = nn.ReplicationPad2d(self.radius)

        # compute patch-ssd

        # shift-align all 4 patch pairings, implemented via convolution
        h1 = F.conv2d(rpad1(img), mshift1, dilation=self.dilation)
        h2 = F.conv2d(rpad1(img), mshift2, dilation=self.dilation)
        # calculate difference 
        diff = rpad2((h1 - h2) ** 2)
        # convolve difference patches via averaging. This makes the loss magnitude invriant of patch size.
        ssd = F.avg_pool2d(diff, kernel_size, stride=1)


        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0] # normalize by substracting lowest value
        mind_var = torch.mean(mind, 1, keepdim=True) # Mean across neighborhood pixel pairings (channels)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item()) # remove outliers
        mind /= mind_var
        mind = torch.exp(-mind)

        return mind

    def forward(self, y_pred, y_true):
        # Get the MIND-SSC descriptor for each image
        if y_pred.dim() == 4:
            true = self.MINDSSC2d(y_true)
            pred = self.MINDSSC2d(y_pred)
        elif y_pred.dim() == 5:
            true = self.MINDSSC3d(y_true)
            pred = self.MINDSSC3d(y_pred)

        # calulate difference
        return torch.mean((true - pred) ** 2)

class NMI(nn.Module):
    """
    Normalized mutual information, using gaussian parzen window estimates.
    Adapted from https://github.com/qiuhuaqi/midir/blob/master/model/loss.py
    """

    def __init__(self,
                 vmin=0.0,
                 vmax=1.0,
                 num_bins=64,
                 sample_ratio=0.1,
                 normalised=True
                 ):
        super().__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.sample_ratio = sample_ratio
        self.normalised = normalised

        # set the std of Gaussian kernel so that FWHM is one bin width
        bin_width = (vmax - vmin) / num_bins
        self.sigma = bin_width * (1/(2 * math.sqrt(2 * math.log(2))))

        # set bin edges
        self.num_bins = num_bins
        self.bins = torch.linspace(
            self.vmin, self.vmax, self.num_bins, requires_grad=False).unsqueeze(1)

    def _compute_joint_prob(self, x, y):
        """
        Compute joint distribution and entropy
        Input shapes (N, 1, prod(sizes))
        """
        # cast bins
        self.bins = self.bins.type_as(x)

        # calculate Parzen window function response (N, #bins, H*W*D)
        win_x = torch.exp(-(x - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_x = win_x / (math.sqrt(2 * math.pi) * self.sigma)
        win_y = torch.exp(-(y - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_y = win_y / (math.sqrt(2 * math.pi) * self.sigma)

        # calculate joint histogram batch
        hist_joint = win_x.bmm(win_y.transpose(1, 2))  # (N, #bins, #bins)

        # normalise joint histogram to get joint distribution
        hist_norm = hist_joint.flatten(
            start_dim=1, end_dim=-1).sum(dim=1) + 1e-5
        # (N, #bins, #bins) / (N, 1, 1)
        p_joint = hist_joint / hist_norm.view(-1, 1, 1)

        return p_joint

    def forward(self, x, y):
        """
        Calculate (Normalised) Mutual Information Loss.
        Args:
            x: (torch.Tensor, size (N, 1, *sizes))
            y: (torch.Tensor, size (N, 1, *sizes))
        Returns:
            (Normalise)MI: (scalar)
        """
        if self.sample_ratio < 1.:
            # random spatial sampling with the same number of pixels/voxels
            # chosen for every sample in the batch
            numel_ = np.prod(x.size()[2:])
            idx_th = int(self.sample_ratio * numel_)
            idx_choice = torch.randperm(int(numel_))[:idx_th]

            x = x.view(x.size()[0], 1, -1)[:, :, idx_choice]
            y = y.view(y.size()[0], 1, -1)[:, :, idx_choice]

        # make sure the sizes are (N, 1, prod(sizes))
        x = x.flatten(start_dim=2, end_dim=-1)
        y = y.flatten(start_dim=2, end_dim=-1)

        # compute joint distribution
        p_joint = self._compute_joint_prob(x, y)

        # marginalise the joint distribution to get marginal distributions
        # batch size in dim0, x bins in dim1, y bins in dim2
        p_x = torch.sum(p_joint, dim=2)
        p_y = torch.sum(p_joint, dim=1)

        # calculate entropy
        ent_x = - torch.sum(p_x * torch.log(p_x + 1e-5), dim=1)  # (N,1)
        ent_y = - torch.sum(p_y * torch.log(p_y + 1e-5), dim=1)  # (N,1)
        ent_joint = - \
            torch.sum(p_joint * torch.log(p_joint + 1e-5), dim=(1, 2))  # (N,1)

        if self.normalised:
            return -torch.mean((ent_x + ent_y) / ent_joint)
        else:
            return -torch.mean(ent_x + ent_y - ent_joint)



if __name__ == "__main__":
    import numpy as np
    import torchreg
    
    import os 

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]='5'
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    torchreg.settings.set_ndims(2)
    H, W, D = 256, 128, 1
    I_m = torch.rand(2,1,H,W,D).squeeze(dim=-1)
    I_1 = torch.rand(2,1,H,W,D).squeeze(dim=-1)
    
    mind_par = 5
    loss2 = MIND_loss(radius=mind_par, dilation=mind_par)(I_m.to('cuda'), I_1.to('cuda'))
    print(loss2)
    print(45*'-')
    #import ipdb; #ipdb.set_trace()


    
