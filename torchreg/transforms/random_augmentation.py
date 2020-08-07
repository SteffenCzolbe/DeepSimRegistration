import torch
import torch.nn as nn
import torch.nn.functional as F
import torchreg.nn as tnn
import torchreg.settings as settings
import numpy as np

class Compose(nn.Module):
    """
    Composes multiple augmentaions

    Args:
        submodules: a list of augmentations
    """
    def __init__(self, submodules):
        self.submodules = submodules

    def randomize(self):
        """
        randomizes the transformations
        """
        for submodule in self.submodules:
            submodule.randomize()

    def forward(self, batch):
        for submodule in self.submodules:
            batch = submodule(batch)
        return batch

class RandomAffine(nn.Module):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to None to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence of float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        flip (boolean), random flips along axis
    """

    def __init__(self, degrees=(-180, 180), translate=(-0.5, 0.5), scale=(0.9, 1.1), shear=(-0.03, 0.03), flip=True):
        super().__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.flip = flip
        self.ndims = settings.get_ndims()

        self.itenditity = tnn.Identity()
        self.transform = tnn.AffineSpatialTransformer()


    def randomize(self):
        """
        randomizes the affine transformation
        """
        if self.degrees is not None:
            rotate = np.random.uniform(*self.degrees, size=self.ndims)
            rotate = np.deg2rad(rotate)
            if self.ndims == 2:
                rotate_matrix = np.array([[np.cos(rotate[0]), -np.sin(rotate[0]), 0],
                                          [np.sin(rotate[0]), np.cos(rotate[0]), 0],
                                          [0, 0, 1]])
            else:
                rx = np.array([[1, 0, 0, 0],
                               [0, np.cos(rotate[0]), -np.sin(rotate[0]), 0],
                               [0, np.sin(rotate[0]), np.cos(rotate[0]), 0],
                               [0, 0, 0, 1]])
                ry = np.array([[np.cos(rotate[1]), 0, np.sin(rotate[1]), 0],
                               [0, 1, 0, 0],
                               [-np.sin(rotate[1]), 0, np.cos(rotate[1]), 0],
                               [0, 0, 0, 1]])
                rz = np.array([[np.cos(rotate[2]), -np.sin(rotate[2]), 0, 0],
                               [np.sin(rotate[2]), np.cos(rotate[2]), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
                rotate_matrix = rx.dot(ry).dot(rz)
        else:
            rotate_matrix = np.eye(self.ndims + 1)

        if self.translate is not None:
            # draw random translate
            translate = np.random.uniform(*self.translate, size=self.ndims)
            # make affine transformation matrix
            translate_matrix = np.eye(self.ndims + 1)
            translate_matrix[:-1, -1] = translate
        else:
            translate_matrix = np.eye(self.ndims + 1)

        if self.scale is not None:
            # draw random scale
            scale = np.random.uniform(*self.scale, size=(self.ndims))
            # add homogenous coordinate
            scale = np.append(scale, [1])
            # make affine transformation matrix
            scale_matrix = np.diag(scale)
        else:
            scale_matrix = np.eye(self.ndims + 1)

        if self.shear is not None:
            # draw random shear
            shear = np.random.uniform(*self.shear, size=(self.ndims, self.ndims))
            shear_matrix = np.eye(self.ndims + 1)
            shear_matrix[:-1, :-1] = shear
            for i in range(self.ndims):
                shear_matrix[i, i] = 1
        else:
            shear_matrix = np.eye(self.ndims + 1)

        if self.flip:
            # draw random flip
            flip = np.sign(np.random.normal(size=self.ndims))
            # add homogenous coordinate
            flip = np.append(flip, [1])
            # make affine transformation matrix
            flip_matrix = np.diag(flip)
        else:
            flip_matrix = np.eye(self.ndims + 1)

        # combine all transformations
        self.affine = rotate_matrix.dot(translate_matrix).dot(scale_matrix).dot(shear_matrix).dot(flip_matrix)
        return

    def apply_affine(self, batch, interpolation, affine):
        # to tensor and expand to batch size
        affine = torch.tensor(affine).type_as(batch)
        B = batch.shape[0]
        affine = affine.repeat(B, *[1]*(self.ndims + 1))
        
        # transform
        return self.transform(batch, affine, mode=interpolation, padding_mode="zeros")

    def apply(self, batch, interpolation='bilinear'):
        assert hasattr(self, "affine"), "The random data augmentation needs to be initialized by calling .randomize()"
        return self.apply_affine(batch, interpolation, self.affine)

    def apply_inverse(self, batch, interpolation='bilinear'):
        assert hasattr(self, "affine"), "The random data augmentation needs to be initialized by calling .randomize()"
        return self.apply_affine(batch, interpolation, np.linalg.inv(self.affine))

    def forward(self, batch, interpolation='bilinear'):
        assert hasattr(self, "affine"), "The random data augmentation needs to be initialized by calling .randomize()"
        return self.apply(batch, interpolation=interpolation)


class RandomDiffeomorphic(nn.Module):
    """
    applies a random diffeomorphic transformation
    This transform is implemented as a nn.Module, and can also be used within the network architecture to be executed on GPU.
    """

    def __init__(self, p=0.5, m=2.0, r=8):
        """
        Parameters:
            p: probability to apply a transform to each image
            m: stddev of the transformations (pixels)
            r: resolution factos on which to generate the random transformation vectors on
        """
        super().__init__()
        self.p = p
        self.m = m
        self.r = r
        self.integrate = tnn.FlowIntegration(nsteps=5)
        self.transform = tnn.SpatialTransformer()

    def randomize(self, size):
        """
        randomizes the transformation
        Parameters:
            size: Size of the data, eg [128, 128, 128]
        """
        self.do_augment = np.random.rand() < self.p
        if not self.do_augment:
            return

        # dimensionality stuff
        C = settings.get_ndims()
        # scale down H,W,D,...
        size_small = [1, C] + [s // self.r for s in size]

        # create some noise at lower resolution
        flow = torch.randn(size_small) * self.m
        # upsample to full resolution
        mode = tnn.interpol_mode("linear")
        flow = F.interpolate(flow, size=size, mode=mode, align_corners=False)
        # integrate for smoothness and invertability
        self.pos_flow = self.integrate(flow)
        self.neg_flow = self.integrate(-flow)

    def apply(self, batch, interpolation='bilinear'):
        """
        apply the transformation, must be previously randomized.
        """
        assert hasattr(self, "do_augment"), "The random data augmentation needs to be initialized by calling .randomize()"
        if not self.do_augment:
            return batch
        return self.transform(batch, self.pos_flow.type_as(batch), mode=interpolation, padding_mode="zeros")

    def apply_inverse(self, batch, interpolation='bilinear'):
        assert hasattr(self, "do_augment"), "The random data augmentation needs to be initialized by calling .randomize()"
        if not self.do_augment:
            return batch
        return self.transform(batch, self.neg_flow.type_as(batch), mode=interpolation, padding_mode="zeros")

    def forward(self, batch, interpolation='bilinear'):
        assert hasattr(self, "do_augment"), "The random data augmentation needs to be initialized by calling .randomize()"
        return self.apply(batch, interpolation=interpolation)


class GaussianNoise(nn.Module):
    """
    Composes multiple augmentaions

    Args:
        std: standard dev of noise
    """
    def __init__(self, std):
        super().__init__()
        self.std = std

    def randomize(self):
        """
        randomizes the transformation
        """
        pass

    def forward(self, batch):
        noise = torch.normal(0, self.std, batch.shape).to(batch.device)
        return batch + noise

class RandomIntenityShift(nn.Module):
    """
    Composes multiple augmentaions

    Args:
        instensity_change_std: std of the magnitude of the effect
        size_std: Std of the filter size
        filter_count: Amount of filter to apply.
        downsize: downsamling factor of the internal computation. Higher numbers speed up the process.
    """
    def __init__(self, instensity_change_std, size_std, filter_count, downsize=16):
        super().__init__()
        self.instensity_change_std = instensity_change_std
        self.size_std = size_std / downsize
        self.filter_count = filter_count
        self.downsize = downsize
        self.identity = tnn.Identity()

    def randomize(self):
        """
        randomizes the transformation
        """
        pass

    def forward(self, batch):
        # create white canvas
        size = torch.tensor(batch.shape[2:]) // self.downsize
        canvas = torch.zeros(1, 1, *size, device = batch.device)

        for _ in range(self.filter_count):
            # set mu and std of effect
            mu = torch.rand(len(size)) * size.float()
            std = torch.normal(mean=self.size_std, std=self.size_std/2, size=(len(size),)).abs()

            # calculate distribution on the canvas
            x = self.identity(canvas).unsqueeze(-1).transpose(1, -1)
            distribution = torch.exp(-torch.sum((x - mu)**2 / std, dim=-1))
            distribution /= torch.max(distribution)

            # upscale and apply to image
            intensity_change = torch.normal(mean=0, std=self.instensity_change_std, size=())
            canvas += intensity_change * distribution
        return batch + F.interpolate(canvas, size=batch.shape[2:], mode='bilinear' if len(size) == 2 else 'trilinear', align_corners=True)
