import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
import torchreg.settings as settings
from .dim_agnostic import interpol_mode

"""
basic spatial transformer layers
"""


class Identity(nn.Module):
    def __init__(self):
        """
        Creates a identity transform
        """
        super().__init__()

    def forward(self, flow):
        # create identity grid
        size = flow.shape[2:]
        vectors = [
            torch.arange(0, s, dtype=flow.dtype, device=flow.device) for s in size
        ]
        grids = torch.meshgrid(vectors)
        identity = torch.stack(grids)  # z, y, x
        identity = identity.expand(flow.shape[0], *[-1] * (settings.get_ndims() + 1))  # add batch
        return identity


class FlowComposition(nn.Module):
    """
    A flow composer, composing two flows /transformations / displacement fields.
    """

    def __init__(self):
        """
        instantiates the FlowComposition
        """
        super().__init__()
        self.transformer = SpatialTransformer()

    def forward(self, *args):
        """
        compose any number of flows
        
        Parameters:
            *args: flows, in order from left to right
        """
        if len(args) == 0:
            raise Exception("Can not compose 0 flows")
        elif len(args) == 1:
            return args[0]
        else:
            composition = self.compose(args[0], args[1])
            return self.forward(composition, *args[2:])

    def compose(self, flow0, flow1):
        """
        compose the flows
        
        Parameters:
            flow0: the first flow
            flow1: the next flow
        """
        return flow0 + self.transformer(flow1, flow0)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, mode="bilinear"):
        """
        Instantiates the spatial transformer. 
        A spatial transformer transforms a src image with a flow of displacement vectors.
        
        Parameters:
            mode: interpolation mode
        """
        super().__init__()
        self.identity = Identity()
        self.grid_sampler = GridSampler(mode=mode)

    def forward(self, src, flow, mode=None, padding_mode="border"):
        """
        Transforms the src with the flow 
        Parameters:
            src: Tensor (B x C x D x H x W)
            flow: Tensor  (B x C x D x H x W) of displacement vextors. Channel 0 indicates the flow in the depth dimension.
            mode: interpolation mode. If not specified, take mode from init function
            padding_mode: 'zeros', 'boarder', 'reflection'
        """

        # map from displacement vectors to absolute coordinates
        coordinates = self.identity(flow) + flow
        return self.grid_sampler(src, coordinates, mode=mode, padding_mode=padding_mode)


class AffineSpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer for affine input
    """

    def __init__(self, mode="bilinear"):
        """
        Instantiates the spatial transformer. 
        A spatial transformer transforms a src image with a flow of displacement vectors.
        
        Parameters:
            mode: interpolation mode
        """
        super().__init__()
        self.identity = Identity()
        self.grid_sampler = GridSampler(mode=mode)
        self.ndims = settings.get_ndims()

    def forward(self, src, affine, mode=None, padding_mode="border"):
        """
        Transforms the src with the flow 
        Parameters:
            src: Tensor (B x C x D x H x W)
            affine: Tensor  (B x 4 x 4) the affine transformation matrix
            mode: interpolation mode. If not specified, take mode from init function
            padding_mode: 'zeros', 'boarder', 'reflection'
        """
        coordinates = self.identity(src)

        # add homogenous coordinate
        coordinates = torch.cat((coordinates, torch.ones(coordinates.shape[0], 1, *coordinates.shape[2:])), dim=1)

        # center the coordinate grid, so that rotation happens around the center of the domain
        size = coordinates.shape[2:]
        for i in range(self.ndims):
            coordinates[:, i] -= size[i] / 2
        
        # permute for batched matric multiplication
        coordinates = coordinates.permute(0,2,3,4,1) if self.ndims ==3 else coordinates.permute(0,2,3,1)
        # we need to do this for each member of the batch separately
        for i in range(len(coordinates)):
            coordinates[i] = torch.matmul(coordinates[i], affine[i])
        coordinates = coordinates.permute(0,-1,1,2,3) if self.ndims ==3 else coordinates.permute(0,-1,1,2)
        # de-homogenize
        coordinates = coordinates[:, :self.ndims]

        # un-center the coordinate grid
        for i in range(self.ndims):
            coordinates[:, i] += size[i] / 2

        return self.grid_sampler(src, coordinates, mode=mode, padding_mode=padding_mode)


class GridSampler(nn.Module):
    """
    A simple Grid sample operation
    """

    def __init__(self, mode="bilinear"):
        """
        Instantiates the grid sampler.
        The grid sampler samples a grid of values at coordinates.
        
        Parameters:
            mode: interpolation mode
        """
        super().__init__()
        self.mode = mode
        self.ndims = settings.get_ndims()

    def forward(self, values, coordinates, mode=None, padding_mode="border"):
        """
        Transforms the src with the flow 
        Parameters:
            src: Tensor (B x C x D x H x W)
            flow: Tensor  (B x C x D x H x W) of displacement vectors. Channel 0 indicates the flow in the depth dimension.
            mode: interpolation mode. If not specified, take mode from init function
            padding_mode: 'zeros', 'boarder', 'reflection'
        """
        mode = mode if mode else self.mode

        # make mode dimentionality-agnostic
        # mode = interpol_mode(mode)

        # clone the coordinate field as we will modift it.
        coordinates = coordinates.clone()
        # normalize coordinates to be within [-1..1]
        size = values.shape[2:]
        for i in range(len(size)):
            coordinates[:, i, ...] = 2 * (coordinates[:, i, ...] / (size[i] - 1) - 0.5)

        # put coordinate channels in last position and
        # reverse channels (in-build pytorch function indexes axis D x H x W and pixel coordinates z,y,x)
        if self.ndims == 2:
            coordinates = coordinates.permute(0, 2, 3, 1)
            coordinates = coordinates[..., [1, 0]]
        elif self.ndims == 3:
            coordinates = coordinates.permute(0, 2, 3, 4, 1)
            coordinates = coordinates[..., [2, 1, 0]]

        # sample
        return nnf.grid_sample(
            values,
            coordinates,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=True,  # align = True is nessesary to behave similar to indexing the transformation.
        )
