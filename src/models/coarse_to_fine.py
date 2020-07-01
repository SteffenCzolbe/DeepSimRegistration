import torch
import torch.nn as nn
import torch.nn.functional as F
import torchreg
import torchreg.nn as tnn
import random

from .backbone import Backbone, FlowPredictor


class CoarseToFineNetwork(nn.Module):
    """
    Coarse-To-Fine network for (unsupervised) nonlinear registration between two images.
    Slightly modified implementation, retuning both the flow and it's inverse.
    """
    def __init__(
        self,
        in_channels,
        enc_feat,
        dec_feat,
        int_steps=5,
        bnorm=True,
        dropout=True,
    ):
        """ 
        Parameters:
            in_channels: channels of the input
            enc_feat: List of encoder filters. e.g. [16, 32, 32, 32]
            dec_feat: List of decoder filters. e.g. [32, 32, 32, 16]
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            bnorm: bool. Perform batch-normalization?
            dropout: bool. Perform dropout?
        """
        super().__init__()

        # configure backbone
        self.backbone = Backbone(
            enc_feat, dec_feat, in_channels=in_channels, dropout=dropout, bnorm=bnorm,
        )
        self.dropout = dropout
        self.integrate = torchreg.nn.FlowIntegration(int_steps)

        # set up pyramidal modules
        self.registration_modules = nn.ModuleList()
        for c in self.backbone.output_channels:
            self.registration_modules.append(
                PyramidRegistrationModule(c)
            )

    def forward(self, source, target):
        """
        Feed a pair of images through the network, predict a transformation
        
        Parameters:
            source: the moving image
            target: the target image
        
        Return:
            displacement: integrated flow
            inv_displacement: inverse displacement
            composed_flow: the flow-field
        """

        # feed both images through the backbone. Separately, with shared weights. 
        # We use the same random seed for both passes (important for dropout).
        seed = random.getrandbits(63)
        torch.manual_seed(seed)
        source_feats = self.backbone(source)
        torch.manual_seed(seed)
        target_feats = self.backbone(target)

        # run through the stages of flow compositions
        composed_flow = None
        stage_flows = []
        for registration_module, source_feat, target_feat in zip(
            self.registration_modules, source_feats, target_feats
        ):
            composed_flow, this_flow = registration_module(
                source_feat, target_feat, composed_flow
            )
            stage_flows.append(this_flow)

        displacement = self.integrate(composed_flow)
        inv_displacement = self.integrate(-composed_flow)
        return displacement, inv_displacement, composed_flow #, stage_flows


class PyramidRegistrationModule(nn.Module):
    """ 
    The Pyramid Registration Module, implementing one layer of the coarse-to-fine transform
    """

    def __init__(self, channels):
        super().__init__()
        """
        Parameters:
            channels: the input channel count
        """
        # transformation of the input
        self.transform = tnn.SpatialTransformer()

        # flow prediction
        self.flow = FlowPredictor(channels * 2)

        # flow composition
        self.compose = tnn.FlowComposition()

        # upsample
        self.upsample = tnn.Upsample(scale_factor=2, mode="linear", align_corners=False)

    def forward(self, x_source, x_target, prev_flow):
        """
        Parameters:
            x_source: feature map of the moving image
            x_target: feature map of the fixed image
            prev_flow: previous flow, None if identity flow.
            
        Return:
            composed_flow: The flow composition up to this stage
            this_flow: the flow of this stage independently
        """
        # transform source features with flow from previous layer
        if prev_flow is not None:
            prev_flow = self.upsample(prev_flow)
            x_source = self.transform(x_source, prev_flow)

        # stack moving and fixed features
        feat = torch.cat([x_source, x_target], dim=1)

        # predict the flow
        this_flow = self.flow(feat)

        # compose new flow with previous one
        if prev_flow is not None:
            composed_flow = self.compose(this_flow, prev_flow)
        else:
            composed_flow = this_flow

        return composed_flow, this_flow
