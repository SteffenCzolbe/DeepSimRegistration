import os
import argparse
import numpy as np
import torch
import torchreg
import torchreg.transforms.functional as ttf
import torchreg.viz as viz
from .common_lightning_model import CommonLightningModel
from .models.AutoEncoder import AutoEncoder


class AutoEncoderModel(CommonLightningModel):
    """
    We use pytorch lightning to organize our model code
    """

    def __init__(self, hparams):
        """
        Init method instantiates the network
        """
        super().__init__(hparams, dataset_image_pairs=False)
        self.hparams = hparams

        # set net
        self.net = AutoEncoder(
            in_channels=self.dataset_config("channels"),
            out_channels=self.dataset_config("channels"),
            enc_feat=self.hparams.channels,
            dec_feat=self.hparams.channels[::-1],
            bnorm=self.hparams.bnorm,
            dropout=self.hparams.dropout,
        )

        self.mse = torch.nn.MSELoss()

    def forward(self, x):
        """
        Same as torch.nn.Module.forward(), however in Lightning you want this to
        define the operations you want to use for prediction (i.e.: on a server or as a feature extractor).
        """
        # run model
        x_pred, _ = self.net(x)
        return x_pred

    def augment(self, x):
        with torch.no_grad():
            # augment
            self.augmentation.randomize()
            x = self.augmentation(x)
        return x

    def _step(self, batch, batch_idx, save_viz=False):
        """
        unified step function.
        """
        # unpack batch
        x, _ = batch

        # augment
        if self.training:
            x = self.augment(x)

        # predict
        x_pred = self.forward(x)

        loss = self.mse(x, x_pred)
        if save_viz:
            self.viz_results(x, x_pred)
        return {
            "loss": loss,
        }

    def viz_results(self, x, x_pred, save=True):
        if self.dataset_config("dataset_type") == "tif":
            # make figure
            fig = viz.Fig(1, 2, f"Epoch {self.current_epoch}", figsize=(8, 4))
            fig.plot_img(0, 0, x[0], vmin=0, vmax=1, title="Input")
            fig.plot_img(0, 1, x_pred[0], vmin=0, vmax=1, title="Reconstruction")

            if save:
                os.makedirs(self.hparams.savedir, exist_ok=True)
                fig.save(
                    os.path.join(self.hparams.savedir, f"{self.current_epoch}.pdf"),
                )
            else:
                return fig

    def extract_features(self, x):
        """
        Extracts deep features from the input.
        Fix parameters before using it as a loss function.
        """
        self.eval()
        feat = []
        for stage in self.net.backbone.encoder:
            x = stage(x)
            feat.append(x)
        return feat

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Adds model specific command-line args
        """
        common_parser = CommonLightningModel.add_common_model_args()
        parser = argparse.ArgumentParser(
            parents=[common_parser, parent_parser], add_help=False
        )
        parser.add_argument(
            "--channels",
            nargs="+",
            type=int,
            default=[64, 128, 256, 512],
            help="U-Net encoder channels. Decoder uses the reverse. Default: [64, 128, 256, 512]",
        )
        parser.add_argument(
            "--bnorm", action="store_true", help="use batchnormalization."
        )
        parser.add_argument("--dropout", action="store_true", help="use dropout")
        parser.add_argument("--savedir", type=str, help="Directory to save images in")
        return parser
