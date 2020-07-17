import os
import argparse
import numpy as np
import torch
import torchreg
import torchreg.viz as viz
from.common_lightning_model import CommonLightningModel
from .models.voxelmorph import Voxelmorph
from .loss_metrics import NCC, DeepSim, VGGFeatureExtractor
from .segmentation_model import SegmentationModel
import ipdb

class RegistrationModel(CommonLightningModel):
    """
    We use pytorch lightning to organize our model code
    """

    def __init__(self, hparams):
        """
        Init method instantiates the network
        """
        super().__init__(hparams, dataset_image_pairs=True)
        self.hparams = hparams

        # set net
        self.net = Voxelmorph(
            in_channels=self.dataset_config('channels'),
            enc_feat=self.hparams.channels,
            dec_feat=self.hparams.channels[::-1],
            bnorm=self.hparams.bnorm,
            dropout=self.hparams.dropout,
        )

        if hparams.loss == 'deepsim':
            if not hparams.deepsim_weights:
                raise ValueError('No weights specified for Deep Similarity Metric.')
            elif hparams.deepsim_weights.lower() == 'vgg':
                feature_extractor = VGGFeatureExtractor()
            else:
                feature_extractor = SegmentationModel.load_from_checkpoint(hparams.deepsim_weights)
            self.deepsim = DeepSim(feature_extractor)

        self.ncc = NCC(window=hparams.ncc_win_size)
        self.mse = torch.nn.MSELoss()
        self.diffusion_reg = torchreg.metrics.GradNorm()
        self.dice_overlap = torchreg.metrics.DiceOverlap(classes=list(range(3)))
        self.transformer = torchreg.nn.SpatialTransformer()

    def forward(self, moving, fixed):
        """
        Same as torch.nn.Module.forward(), however in Lightning you want this to
        define the operations you want to use for prediction (i.e.: on a server or as a feature extractor).
        """
        # run model
        return self.net(moving, fixed)

    def similarity_loss(self, I_m, I_1, S_m_onehot, S_1_onehot):
        if self.hparams.loss == 'deepsim':
            return self.deepsim(I_m, I_1)
        elif self.hparams.loss == 'ncc':
            return self.ncc(I_m, I_1)
        elif self.hparams.loss == 'ncc+supervised':
            return self.ncc(I_m, I_1) + self.mse(S_m_onehot, S_1_onehot)
        else:
            raise ValueError(f'loss function "{self.hparams.loss}" unknow.')

    def _step(self, batch, batch_idx, save_viz=False):
        """
        unified step function.
        """
        # unpack batch
        (I_0, S_0), (I_1, S_1) = batch


        if self.training:
            with torch.no_grad():
                # augment
                self.augmentation.randomize()
                I_0 = self.augmentation(I_0)
                I_1 = self.augmentation(I_1)
                S_0 = self.augmentation(S_0.float(), interpolation='nearest').round().long()
                S_1 = self.augmentation(S_1.float(), interpolation='nearest').round().long()

        # predict flow
        flow = self.forward(I_0, I_1)

        # morph image and segmentation
        I_m = self.transformer(I_0, flow)
        S_m = self.transformer(S_0.float(), flow, mode='nearest').round().long()
        S_0_onehot = torch.nn.functional.one_hot(S_0[:, 0], num_classes=3).permute(0, -1, 1, 2).float()
        S_m_onehot = self.transformer(S_0_onehot, flow)
        S_1_onehot = torch.nn.functional.one_hot(S_1[:, 0], num_classes=3).permute(0, -1, 1, 2).float()

        # calculate loss
        similarity_loss = self.similarity_loss(I_m, I_1, S_m_onehot, S_1_onehot)
        diffusion_regularization = self.diffusion_reg(flow)
        loss = similarity_loss + self.hparams.lam * diffusion_regularization

        # calculate other (supervised) evaluation mesures
        with torch.no_grad():
            dice_overlap = self.dice_overlap(S_m, S_1)
            accuracy = torch.mean((S_m == S_1).float())

        # visualize
        if save_viz:
            self.viz_results(I_0, I_m, I_1, S_0, S_m, S_1, flow)

        return {
            "loss": loss,
            "regularization": diffusion_regularization,
            "similarity_loss": similarity_loss,
            "dice_overlap": dice_overlap,
            "accuracy": accuracy,
        }

    def viz_results(self, I_0, I_m, I_1, S_0, S_m, S_1, flow, save=True):
        # make figure
        fig = viz.Fig(2, 3, f"Epoch {self.current_epoch}", figsize=(9, 6))

        fig.plot_img(0, 0, I_0[0], vmin=0, vmax=1, title="$I_0$")
        fig.plot_overlay_class_mask(0, 0, S_0[0], num_classes=self.dataset_config('classes'), 
            colors=self.dataset_config('class_colors'), alpha=0.2)

        fig.plot_img(0, 1, I_m[0], vmin=0, vmax=1, title="$I_0 \circ \Phi$")
        fig.plot_overlay_class_mask(0, 1, S_m[0], num_classes=self.dataset_config('classes'), 
            colors=self.dataset_config('class_colors'), alpha=0.2)

        fig.plot_img(1, 1, I_1[0], vmin=0, vmax=1, title="$I_1$")
        fig.plot_overlay_class_mask(1, 1, S_1[0], num_classes=self.dataset_config('classes'), 
            colors=self.dataset_config('class_colors'), alpha=0.2)

        fig.plot_transform_grid(
            1, 0, flow[0], title="$\Phi$", interval=15, linewidth=0.1
        )
        fig.plot_img(0, 2, (S_0[0] != S_1[0]).long(), vmin=0, vmax=1, title="Diff")
        fig.plot_img(1, 2, (S_m[0] != S_1[0]).long(), vmin=0, vmax=1, title="Diff Registered")

        if save:
            os.makedirs(self.hparams.savedir, exist_ok=True)
            fig.save(os.path.join(self.hparams.savedir, f"{self.current_epoch}.pdf"),)
        else:
            return fig

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Adds model specific command-line args
        """
        common_parser = CommonLightningModel.add_common_model_args()
        parser = argparse.ArgumentParser(parents=[common_parser, parent_parser], add_help=False)
        parser.add_argument(
            "--loss", type=str, default='ncc', help="Similarity Loss function. Options: 'ncc', 'deepsim', 'ncc+supervised' (Default: ncc)"
        )
        parser.add_argument(
            "--ncc_win_size", type=int, default=9, help="Window-Size for the NCC loss function (Default: 9)"
            )
        parser.add_argument(
            "--deepsim_weights", type=str, default=None, help="Path to deep feature model weights, OR: 'vgg'"
        )
        parser.add_argument(
            "--lam", type=float, default=0.5, help="Diffusion regularizer strength"
        )
        parser.add_argument(
            "--channels", nargs='+', type=int, default=[64, 128, 256, 512], help="U-Net encoder channels. Decoder uses the reverse. Defaukt: [64, 128, 256, 512]"
        )
        parser.add_argument(
            "--bnorm", action='store_true', help="use batchnormalization."
        )
        parser.add_argument(
            "--dropout", action='store_true', help="use dropout"
        )
        parser.add_argument(
            "--savedir", type=str, help="Directory to save images in"
        )
        return parser
