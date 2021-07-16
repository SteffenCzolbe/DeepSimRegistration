import os
import argparse
import numpy as np
import torch
import torchreg
import torchreg.viz as viz
from .common_lightning_model import CommonLightningModel
from .models.voxelmorph import Voxelmorph
from .loss_metrics import NCC, DeepSim, VGGFeatureExtractor
from .segmentation_model import SegmentationModel
from .autoencoder_model import AutoEncoderModel


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
        self.probabilistic= False
        self.probabilistic_p = 0.5

        # set net
        self.net = Voxelmorph(
            in_channels=self.dataset_config("channels"),
            enc_feat=self.hparams.channels,
            dec_feat=self.hparams.channels[::-1],
            bnorm=self.hparams.bnorm,
            dropout=self.hparams.dropout,
        )
        
        if hparams.loss.lower() in ["deepsim", "deepsim-transfer", "deepsim-ae"] and not hparams.deepsim_weights:
            raise ValueError("No weights specified for Deep Similarity Metric.")
        
        if hparams.loss.lower() in ["deepsim", "deepsim-transfer"]:
            feature_extractor = SegmentationModel.load_from_checkpoint(
                hparams.deepsim_weights
            )
            self.deepsim = DeepSim(feature_extractor)
        elif hparams.loss.lower() == "deepsim-ae":
            feature_extractor = AutoEncoderModel.load_from_checkpoint(
                hparams.deepsim_weights
            )
            self.deepsim = DeepSim(feature_extractor)
        elif hparams.loss.lower() == "vgg":
            feature_extractor = VGGFeatureExtractor()
            self.vgg_loss = DeepSim(feature_extractor)

        squared_ncc = hparams.loss.lower() in ["ncc2", "ncc2+supervised"]
        self.ncc = NCC(window=hparams.ncc_win_size, squared=squared_ncc)
        self.mse = torch.nn.MSELoss()
        self.diffusion_reg = torchreg.metrics.GradNorm()
        self.jacobian_determinant = torchreg.metrics.JacobianDeterminant(reduction='none')
        self.dice_overlap = torchreg.metrics.DiceOverlap(
            classes=list(range(self.dataset_config("classes")))
        )
        self.dice_overlap_per_class = torchreg.metrics.DiceOverlap(
            classes=list(range(self.dataset_config("classes"))), mean_over_classes=False
        )
        self.transformer = torchreg.nn.SpatialTransformer()

    def forward(self, moving, fixed):
        """
        Same as torch.nn.Module.forward(), however in Lightning you want this to
        define the operations you want to use for prediction (i.e.: on a server or as a feature extractor).
        """
            
        # activate dropout layers for probabilistic model
        if self.probabilistic:
            dropout_layers = self.get_dropout_layers(self)
            for l in dropout_layers:
                l.train() # activate dropout
                l.p = self.probabilistic_p # set dropout probability
                
        # run model
        return self.net(moving, fixed)

    def similarity_loss(self, I_m, I_1, S_m_onehot, S_1_onehot):
        if self.hparams.loss.lower() in ["deepsim", "deepsim-transfer", "deepsim-ae"]:
            return self.deepsim(I_m, I_1)
        elif self.hparams.loss.lower() == "vgg":
            return self.vgg_loss(I_m, I_1)
        elif self.hparams.loss.lower() in ["ncc", "ncc2"]:
            return self.ncc(I_m, I_1)
        elif self.hparams.loss.lower() in ["ncc+supervised", "ncc2+supervised"]:
            return self.ncc(I_m, I_1) + self.mse(S_m_onehot, S_1_onehot)
        elif self.hparams.loss.lower() == "l2":
            return self.mse(I_m, I_1)
        else:
            raise ValueError(f'loss function "{self.hparams.loss}" unknow.')

    def augment(self, I_0, I_1, S_0, S_1):
        with torch.no_grad():
            self.augmentation.randomize()
            I_0 = self.augmentation(I_0)
            I_1 = self.augmentation(I_1)
            S_0 = self.augmentation(S_0.float(), interpolation="nearest").round().long()
            S_1 = self.augmentation(S_1.float(), interpolation="nearest").round().long()
        return I_0, I_1, S_0, S_1

    def segmentation_to_onehot(self, S):
        return (
            torch.nn.functional.one_hot(
                S[:, 0], num_classes=self.dataset_config("classes")
            )
            .unsqueeze(1)
            .transpose(1, -1)
            .squeeze(-1)
            .float()
        )
        
    
    def get_dropout_layers(self, model):
        """
        Collects all the dropout layers of the model
        """
        ret = []
        for obj in model.children():
            if hasattr(obj, 'children'):
                ret += self.get_dropout_layers(obj)
            if isinstance(obj, torch.nn.Dropout3d) or isinstance(obj, torch.nn.Dropout2d):
                ret.append(obj)
        return ret

    def _step(self, batch, batch_idx, save_viz=False, eval_per_class=False):
        """
        unified step function.

        Parameters:
            batch: the batch
            batch_idx: int, batch index no (unused)
            save_viz: save a visualization of this batch. Default False.
            eval_per_class: add a dice-overlap score per class. Default False.
        """
        # unpack batch
        (I_0, S_0), (I_1, S_1) = batch

        # augment
        if self.training:
            I_0, I_1, S_0, S_1 = self.augment(I_0, I_1, S_0, S_1)

        # predict flow
        flow = self.forward(I_0, I_1)

        # morph image and segmentation
        I_m = self.transformer(I_0, flow)
        S_m = self.transformer(S_0.float(), flow, mode="nearest").round().long()
        S_0_onehot = self.segmentation_to_onehot(S_0)
        S_m_onehot = self.transformer(S_0_onehot, flow)
        S_1_onehot = self.segmentation_to_onehot(S_1)

        # calculate loss
        similarity_loss = self.similarity_loss(I_m, I_1, S_m_onehot, S_1_onehot)
        diffusion_regularization = self.diffusion_reg(flow)
        loss = similarity_loss + self.hparams.lam * diffusion_regularization

        # calculate other (supervised) evaluation mesures
        with torch.no_grad():
            dice_overlap = self.dice_overlap(S_m, S_1)
            accuracy = torch.mean((S_m == S_1).float())
            if eval_per_class:
                dice_overlap_per_class = self.dice_overlap_per_class(S_m, S_1)
            # jacobian determinants
            jac_dets = self.jacobian_determinant(flow)

        # visualize
        if save_viz and self.dataset_config("dataset_type") == "tif":
            self.viz_results(I_0, I_m, I_1, S_0, S_m, S_1, flow)

        return {
            "loss": loss,
            "regularization": diffusion_regularization,
            "similarity_loss": similarity_loss,
            "dice_overlap": dice_overlap,
            "accuracy": accuracy,
            "jacobian_determinant_mean": jac_dets.mean(),
            "jacobian_determinant_negative": (jac_dets < 0).float().mean(),
            "jacobian_determinant_var": jac_dets.var(),
            "jacobian_determinant_log_var": jac_dets.abs().log().var(),
            "dice_overlap_per_class": dice_overlap_per_class
            if eval_per_class
            else None,
        }

    def viz_results(self, I_0, I_m, I_1, S_0, S_m, S_1, flow, save=True):
        # make figure
        fig = viz.Fig(2, 3, f"Epoch {self.current_epoch}", figsize=(9, 6))

        fig.plot_img(0, 0, I_0[0], vmin=0, vmax=1, title="$I_0$")
        fig.plot_overlay_class_mask(
            0,
            0,
            S_0[0],
            num_classes=self.dataset_config("classes"),
            colors=self.dataset_config("class_colors"),
            alpha=0.2,
        )

        fig.plot_img(0, 1, I_m[0], vmin=0, vmax=1, title="$I_0 \circ \Phi$")
        fig.plot_overlay_class_mask(
            0,
            1,
            S_m[0],
            num_classes=self.dataset_config("classes"),
            colors=self.dataset_config("class_colors"),
            alpha=0.2,
        )

        fig.plot_img(1, 1, I_1[0], vmin=0, vmax=1, title="$I_1$")
        fig.plot_overlay_class_mask(
            1,
            1,
            S_1[0],
            num_classes=self.dataset_config("classes"),
            colors=self.dataset_config("class_colors"),
            alpha=0.2,
        )

        fig.plot_transform_grid(
            1, 0, flow[0], title="$\Phi$", interval=15, linewidth=0.1
        )
        fig.plot_img(0, 2, (S_0[0] != S_1[0]).long(), vmin=0, vmax=1, title="Diff")
        fig.plot_img(
            1, 2, (S_m[0] != S_1[0]).long(), vmin=0, vmax=1, title="Diff Registered"
        )

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
        parser = argparse.ArgumentParser(
            parents=[common_parser, parent_parser], add_help=False
        )
        parser.add_argument(
            "--loss",
            type=str,
            default="ncc",
            help="Similarity Loss function. Options: 'l2', 'ncc', 'ncc2', 'deepsim', 'deepsim-transfer', 'deepsim-ae', 'ncc+supervised', 'vgg' (Default: ncc)",
        )
        parser.add_argument(
            "--ncc_win_size",
            type=int,
            default=9,
            help="Window-Size for the NCC loss function (Default: 9)",
        )
        parser.add_argument(
            "--deepsim_weights",
            type=str,
            default=None,
            help="Path to deep feature model weights, required for loss='deepsim' and 'deepsim-transfer'",
        )
        parser.add_argument(
            "--lam", type=float, default=0.5, help="Diffusion regularizer strength"
        )
        parser.add_argument(
            "--channels",
            nargs="+",
            type=int,
            default=[64, 128, 256, 512],
            help="U-Net encoder channels. Decoder uses the reverse. Defaukt: [64, 128, 256, 512]",
        )
        parser.add_argument(
            "--bnorm", action="store_true", help="use batchnormalization."
        )
        parser.add_argument("--dropout", action="store_true", help="use dropout")
        parser.add_argument("--savedir", type=str, help="Directory to save images in")
        return parser
