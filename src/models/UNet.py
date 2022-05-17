import torch
import torch.nn as nn
import torch.nn.functional as F
import torchreg.nn as tnn

from .backbone import Backbone


class UNet(nn.Module):
    """
    UNet network for classification.
    """

    def __init__(
        self, in_channels, out_channels, enc_feat, dec_feat, bnorm=False, dropout=True
    ):
        """ 
        Parameters:
            in_channels: channels of the input
            out_channels: channels of the output
            enc_feat: List of encoder filters. e.g. [16, 32, 32, 32]
            dec_feat: List of decoder filters. e.g. [32, 32, 32, 16]
            bnorm: bool. Perform batch-normalization?
            dropout: bool. Perform dropout?
        """
        super().__init__()

        # configure backbone
        self.backbone = Backbone(
            enc_feat, dec_feat, in_channels=in_channels, dropout=dropout, bnorm=bnorm,
        )

        # configure output layer
        self.pred = nn.Sequential(
            tnn.Conv(
                self.backbone.output_channels[-1],
                self.backbone.output_channels[-1],
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
            tnn.Conv(
                self.backbone.output_channels[-1],
                self.backbone.output_channels[-1],
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
            tnn.Conv(
                self.backbone.output_channels[-1],
                out_channels,
                kernel_size=1,
                padding=0,
            ),
        )
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Feed a pair of images through the network, predict a segmentation
        
        Parameters:
            x: the intensity image
        Returns:
            y_pred: the normalized predictions, summing up to 1 over classes
            y_pred_raw: raw numerical predictions
        """
        # feed through network
        feat = self.backbone(x)
        y_pred_raw = self.pred(feat[-1])
        y_pred = self.activation(y_pred_raw)

        return y_pred, y_pred_raw
