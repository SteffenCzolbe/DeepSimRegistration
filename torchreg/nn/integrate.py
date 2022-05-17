import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
from .layers import SpatialTransformer
from .dim_agnostic import interpol_mode


class FlowIntegration(nn.Module):
    """
    Integrates a displacement vector field via scaling and squaring.
    """

    def __init__(self, nsteps, downsize=1):
        """ 
        Parameters:
            nsteps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
        """
        super().__init__()

        assert nsteps >= 0, "nsteps should be >= 0, found: %d" % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer()

        # configure optional resize layers
        resize = downsize > 1
        self.resize = ResizeTransform(downsize) if resize else None
        self.fullsize = ResizeTransform(1.0 / downsize) if resize else None

    def forward(self, flow):
        # resize
        if self.resize:
            flow = self.resize(flow)

        # scaling ...
        flow = flow * self.scale

        # and squaring ...
        for _ in range(self.nsteps):
            flow = flow + self.transformer(flow, flow)

        # resize back to full size
        if self.fullsize:
            flow = self.fullsize(flow)
        return flow


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = interpol_mode("linear")

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x
