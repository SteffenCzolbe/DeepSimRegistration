import torch
import torch.nn as nn
import torch.nn.functional as F
import torchreg.nn as tnn

from .backbone import Backbone, FlowPredictor


class Voxelmorph(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    Slightly modified implementation.
    """
    def __init__(
        self,
        in_channels,
        enc_feat,
        dec_feat,
        bnorm=True,
        dropout=True,
    ):
        """ 
        Parameters:
            in_channels: channels of the input
            enc_feat: List of encoder filters. e.g. [16, 32, 32, 32]
            dec_feat: List of decoder filters. e.g. [32, 32, 32, 16]
            bnorm: bool. Perform batch-normalization?
            dropout: bool. Perform dropout?
        """
        super().__init__()

        # configure backbone
        self.backbone = Backbone(
            enc_feat,
            dec_feat,
            in_channels=2 * in_channels,
            dropout=dropout,
            bnorm=bnorm,
        )

        # configure flow prediction and integration
        self.flow = FlowPredictor(
            in_channels=self.backbone.output_channels[-1],
        )

    def forward(self, source, target):
        """
        Feed a pair of images through the network, predict a transformation
        
        Parameters:
            source: the moving image
            target: the target image
        
        Return:
            the flow
        """
        # concatenate inputs
        x = torch.cat([source, target], dim=1)

        # feed through network
        dec_activations = self.backbone(x)
        x = dec_activations[-1]

        # predict flow and integrate
        flow = self.flow(x)

        return flow
