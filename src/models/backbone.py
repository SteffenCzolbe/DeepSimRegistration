import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torchreg.nn as tnn
import torchreg.settings as settings
import numpy as np


class Backbone(nn.Module):
    """ 
    U-net backbone for registration models.
    """

    def __init__(self, enc_feat, dec_feat, in_channels=1, bnorm=False, dropout=True):
        """
        Parameters:
            enc_feat: List of encoder features. e.g. [16, 32, 32, 32]
            dec_feat: List of decoder features. e.g. [32, 32, 32, 16]
            in_channels: input channels, eg 1 for a single greyscale image. Default 1.
            bnorm: bool. Perform batch-normalization?
            dropout: bool. Perform dropout?
        """
        super().__init__()

        self.upsample = tnn.Upsample(scale_factor=2, mode="linear", align_corners=False)

        # configure encoder (down-sampling path)
        prev_feat = in_channels
        self.encoder = nn.ModuleList()
        for feat in enc_feat:
            self.encoder.append(
                Stage(prev_feat, feat, stride=2, dropout=dropout, bnorm=bnorm)
            )
            prev_feat = feat

        # pre-calculate decoder sizes and channels
        enc_stages = len(enc_feat)
        dec_stages = len(dec_feat)
        enc_history = list(reversed([in_channels] + enc_feat))
        decoder_out_channels = [
            enc_history[i + 1] + dec_feat[i] for i in range(dec_stages)
        ]
        decoder_in_channels = [enc_history[0]] + decoder_out_channels[:-1]

        # pre-calculate return sizes and channels
        self.output_length = len(dec_feat) + 1
        self.output_channels = [enc_history[0]] + decoder_out_channels

        # configure decoder (up-sampling path)
        self.decoder = nn.ModuleList()
        for i, feat in enumerate(dec_feat):
            self.decoder.append(
                Stage(
                    decoder_in_channels[i], feat, stride=1, dropout=dropout, bnorm=False
                )
            )

    def forward(self, x):
        """
        Feed x throught the U-Net
        
        Parameters:
            x: the input
        
        Return:
            list of decoder activations, from coarse to fine. Last index is the full resolution output.
        """
        # pass through encoder, save activations
        x_enc = [x]
        for layer in self.encoder:
            x_enc.append(layer(x_enc[-1]))

        # pass through decoder
        x = x_enc.pop()
        x_dec = [x]
        for layer in self.decoder:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)
            x_dec.append(x)

        return x_dec


class Stage(nn.Module):
    """
    Specific U-net stage
    """

    def __init__(self, in_channels, out_channels, stride=1, bnorm=True, dropout=True):
        super().__init__()

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise ValueError("stride must be 1 or 2")

        # build stage
        layers = []
        if bnorm:
            layers.append(tnn.BatchNorm(in_channels))
        layers.append(tnn.Conv(in_channels, out_channels, ksize, stride, 1))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(tnn.Conv(out_channels, out_channels, 3, 1, 1))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(tnn.Dropout())

        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)


class FlowPredictor(nn.Module):
    """
    A layer intended for flow prediction. Initialied with small weights for faster training.
    """

    def __init__(self, in_channels):
        super().__init__()
        """
        instantiates the flow prediction layer.
        
        Parameters:
            in_channels: input channels
        """
        ndims = settings.get_ndims()
        # configure cnn
        self.cnn = nn.Sequential(
            tnn.Conv(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            tnn.Conv(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            tnn.Conv(in_channels, ndims, kernel_size=3, padding=1),
        )

        # init final cnn layer with small weights and bias
        self.cnn[-1].weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.cnn[-1].weight.shape)
        )
        self.cnn[-1].bias = nn.Parameter(torch.zeros(self.cnn[-1].bias.shape))

    def forward(self, x):
        """
        predicts the transformation. 
        
        Parameters:
            x: the input
            
        Return:
            pos_flow, neg_flow: the positive and negative flow
        """
        # predict the flow
        return self.cnn(x)
