import os
import argparse
import numpy as np
import torch
import torchreg
import torchreg.transforms.functional as ttf
import torchreg.viz as viz
from.common_lightning_model import CommonLightningModel
from .models.UNet import UNet
import ipdb

class SegmentationModel(CommonLightningModel):
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
        self.net = UNet(
            in_channels=self.dataset_config('channels'),
            out_channels=self.dataset_config('classes'),
            enc_feat=self.hparams.channels,
            dec_feat=self.hparams.channels[::-1],
            bnorm=self.hparams.bnorm,
            dropout=self.hparams.dropout,
        )

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.dice_overlap = torchreg.metrics.DiceOverlap(classes=list(range(self.dataset_config('classes'))))

    def forward(self, x):
        """
        Same as torch.nn.Module.forward(), however in Lightning you want this to
        define the operations you want to use for prediction (i.e.: on a server or as a feature extractor).
        """
        # run model
        y_pred_onehot, y_pred_raw = self.net(x)
        y_pred = torch.argmax(y_pred_onehot, dim=1, keepdim=True)
        return y_pred, y_pred_raw

    def _step(self, batch, batch_idx, save_viz=False):
        """
        unified step function.
        """
        # unpack batch
        x, y_true = batch

        if self.training:
            with torch.no_grad():
                # augment
                self.augmentation.randomize()
                x = self.augmentation(x)
                y_true = self.augmentation(y_true.float(), interpolation='nearest').round().long()

        y_pred, y_pred_raw = self.forward(x)

        loss = self.cross_entropy_loss(y_pred_raw, y_true.squeeze(1))
        dice_overlap = self.dice_overlap(y_true, y_pred)
        accuracy = torch.mean((y_true == y_pred).float())
        if save_viz:
            self.viz_results(x, y_true, y_pred)
        return {
            "loss": loss,
            "dice_overlap": dice_overlap,
            "accuracy": accuracy,
        }

    def viz_results(self, x, y_true, y_pred, save=True):
        if self.dataset_config('dataset_type') == 'tif':
            # make figure
            fig = viz.Fig(1, 3, f"Epoch {self.current_epoch}", figsize=(8, 3))
            fig.plot_img(0, 0, x[0], vmin=0, vmax=1, title="Input")
            fig.plot_img(0, 1, x[0], title="Prediction")
            fig.plot_overlay_class_mask(0, 1, y_pred[0], num_classes=self.dataset_config('classes'), 
                colors=self.dataset_config('class_colors'), alpha=0.5)
            fig.plot_img(0, 2, x[0], title="Ground Truth")
            fig.plot_overlay_class_mask(0, 2, y_true[0], num_classes=self.dataset_config('classes'), 
                colors=self.dataset_config('class_colors'), alpha=0.5)

            if save:
                os.makedirs(self.hparams.savedir, exist_ok=True)
                fig.save(os.path.join(self.hparams.savedir, f"{self.current_epoch}.pdf"),)
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
        parser = argparse.ArgumentParser(parents=[common_parser, parent_parser], add_help=False)
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
