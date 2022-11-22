import os
import argparse
import numpy as np
import torch
import torchreg
import torchreg.viz as viz
from .common_lightning_model import CommonLightningModel
from .models.voxelmorph import Voxelmorph
from .loss_metrics import NCC, DeepSim, DeepSim_v2, VGGFeatureExtractor, NMI, MIND_loss
from .segmentation_model import SegmentationModel
from .autoencoder_model import AutoEncoderModel

from src.models.TransMorph2D import TransMorph2D, CONFIGS2D 
import torch.nn.functional as F

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
        
        # backwards compatibilty
        if "net" not in self.hparams:
            self.hparams.net = 'voxelmorph'

        # set net
        if self.hparams.net == 'voxelmorph':
            self.net = Voxelmorph(
                in_channels=self.dataset_config("channels"),
                enc_feat=self.hparams.channels,
                dec_feat=self.hparams.channels[::-1],
                bnorm=self.hparams.bnorm,
                dropout=self.hparams.dropout,
            )
        elif self.hparams.net.lower() == 'transmorph2d-small' or self.hparams.net.lower() == 'transmorph2d':
            model_name = self.hparams.net.lower() + '_' + self.hparams.dataset.lower()
            config2D = CONFIGS2D[model_name]
            self.net = TransMorph2D(config2D)
        else:
            raise ValueError(f'model "{self.hparams.net}" unknow.')
        
        if hparams.loss.lower() in ["deepsim", "deepsim-transfer", "deepsim-ae", "deepsim-transfer-ae",
                                    "deepsim-zero", "deepsim-ae-zero",
                                    "deepsim-ae_0", "deepsim-ae_01", "deepsim-ae_02",
                                    "deepsim-ae_1", "deepsim-ae_12", "deepsim-ae_2",
                                    "deepsim_0", "deepsim_01", "deepsim_02",
                                    "deepsim_1", "deepsim_12", "deepsim_2",
                                    "deepsim-ebw", "deepsim-ae-ebw"] and not hparams.deepsim_weights:
            raise ValueError("No weights specified for Deep Similarity Metric.")
        
        if hparams.loss.lower() in ["deepsim", "deepsim-transfer", "deepsim-zero", 
                                    "deepsim_0", "deepsim_01", "deepsim_02",
                                    "deepsim_1", "deepsim_12", "deepsim_2",
                                    "deepsim-ebw"]:

            if hparams.loss.lower()=="deepsim_0":
                levels = [0]
            elif hparams.loss.lower()=="deepsim_1":
                levels = [1]
            elif hparams.loss.lower()=="deepsim_2":
                levels = [2]
            elif hparams.loss.lower()=="deepsim_01":
                levels = [0,1]
            elif hparams.loss.lower()=="deepsim_12":
                levels = [1,2]
            elif hparams.loss.lower()=="deepsim_02":
                levels = [0, 2]
            elif hparams.loss.lower()=="deepsim":
                levels = 'all'
            elif hparams.loss.lower()=="deepsim-transfer":
                levels = 'all'
            elif hparams.loss.lower()=="deepsim-zero":
                levels = 'all'
   

            feature_extractor = SegmentationModel.load_from_checkpoint(
                hparams.deepsim_weights
            )

            if 'zero' in hparams.loss.lower():
                self.deepsim = DeepSim(feature_extractor, levels=levels, zero_mean=True)
            elif 'ebw' in hparams.loss.lower():
                    self.deepsim = DeepSim_v2(feature_extractor)
            else:
                self.deepsim = DeepSim(feature_extractor, levels=levels, zero_mean=False)

        elif hparams.loss.lower() in ["deepsim-ae", "deepsim-transfer-ae","deepsim-ae-zero",
                                      "deepsim-ae_0", "deepsim-ae_01", "deepsim-ae_02",
                                      "deepsim-ae_1", "deepsim-ae_12", "deepsim-ae_2",
                                      "deepsim-ae-ebw"]:

            if hparams.loss.lower()=="deepsim-ae_0":
                levels = [0]
            elif hparams.loss.lower()=="deepsim-ae_1":
                levels = [1]
            elif hparams.loss.lower()=="deepsim-ae_2":
                levels = [2]
            elif hparams.loss.lower()=="deepsim-ae_01":
                levels = [0,1]
            elif hparams.loss.lower()=="deepsim-ae_12":
                levels = [1,2]
            elif hparams.loss.lower()=="deepsim-ae_02":
                levels = [0, 2]
            elif hparams.loss.lower()=="deepsim-ae":
                levels = 'all'
            elif hparams.loss.lower()=="deepsim-transfer-ae":
                levels = 'all'
            elif hparams.loss.lower()=="deepsim-ae-zero":
                levels = 'all'

            feature_extractor = AutoEncoderModel.load_from_checkpoint(
                hparams.deepsim_weights
            )

            if 'zero' in hparams.loss.lower():
                self.deepsim = DeepSim(feature_extractor, levels=levels, zero_mean=True)
            elif 'ebw' in hparams.loss.lower():
                    self.deepsim = DeepSim_v2(feature_extractor)
            else:
                self.deepsim = DeepSim(feature_extractor, levels=levels, zero_mean=False)

        elif hparams.loss.lower() == "vgg":
            feature_extractor = VGGFeatureExtractor()
            self.vgg_loss = DeepSim(feature_extractor)
        elif hparams.loss.lower() == "mind":
            self.mind_loss = MIND_loss()
        elif hparams.loss.lower() == "nmi":
            # brain dataset hyperintensities can be >1.0
            self.nmi_loss = NMI(
                vmax=1.5 if hparams.dataset == "brain-mri" else 1.,
                num_bins=hparams.nmi_bin_size)

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

        # register the images
        if self.hparams.net.lower() == 'voxelmorph':
            # activate dropout layers for probabilistic model
            if self.probabilistic:
                dropout_layers = self.get_dropout_layers(self)
                for l in dropout_layers:
                    l.train() # activate dropout
                    l.p = self.probabilistic_p # set dropout probability
            flow = self.net(moving, fixed)
        else:
            x = torch.cat((moving, fixed), dim=1)
            flow = self.net(x)
            #moved, flow = self.net(x)

        return flow

    def similarity_loss(self, I_m, I_1, S_m_onehot, S_1_onehot):
        if self.hparams.loss.lower() in ["deepsim", "deepsim-transfer", "deepsim-ae", "deepsim-transfer-ae",
                                         "deepsim-zero", "deepsim-ae-zero",
                                         "deepsim-ae_0", "deepsim-ae_01", "deepsim-ae_02",
                                         "deepsim-ae_1", "deepsim-ae_12", "deepsim-ae_2",
                                         "deepsim_0", "deepsim_01", "deepsim_02",
                                         "deepsim_1", "deepsim_12", "deepsim_2",
                                         "deepsim-ebw", "deepsim-ae-ebw"]:
            return self.deepsim(I_m, I_1)
        elif self.hparams.loss.lower() in ["deepsim-ebw", "deepsim-ae-ebw"]:
            # extract before warp
            self.deepsim.first_extract_features_then_warp(I_0, I_1, flow)
        elif self.hparams.loss.lower() == "vgg":
            return self.vgg_loss(I_m, I_1)
        elif self.hparams.loss.lower() in ["ncc", "ncc2"]:
            return self.ncc(I_m, I_1)
        elif self.hparams.loss.lower() in ["ncc+supervised", "ncc2+supervised"]:
            return self.ncc(I_m, I_1) + self.mse(S_m_onehot, S_1_onehot)
        elif self.hparams.loss.lower() == "l2":
            return self.mse(I_m, I_1)
        elif self.hparams.loss.lower() == "mind":
            if self.hparams.dataset.lower() in ["phc-u373", "platelet-em"]:
                return self.mind_loss(I_m.unsqueeze(dim=-1), I_1.unsqueeze(dim=-1))
            else:
                return self.mind_loss(I_m, I_1) 
        elif self.hparams.loss.lower() == "nmi":
            return self.nmi_loss(I_m, I_1)
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

        if self.hparams.dataset.lower() == "phc-u373":
            if 'transmorph2d' in self.hparams.net.lower():
                I_0 = F.pad(I_0, (8, 8) , "constant", 0)
                I_1 = F.pad(I_1, (8, 8) , "constant", 0)
                S_0 = F.pad(S_0, (8, 8) , "constant", 0)
                S_1 = F.pad(S_1, (8, 8) , "constant", 0)

        # augment
        if self.training:
            I_0, I_1, S_0, S_1 = self.augment(I_0, I_1, S_0, S_1)
        
        # predict flow
        #flow = self.forward(I_0, I_1)
        if self.hparams.net.lower() == 'voxelmorph':
            flow = self.net(I_0, I_1)
        else:
            x = torch.cat((I_0, I_1), dim=1)
            flow = self.net(x)
            #moved, flow = self.net(x)
        
        if self.hparams.dataset.lower() == "phc-u373":
            if 'transmorph2d' in self.hparams.net.lower():
                I_0 = I_0[:,:,:,8:-8]
                I_1 = I_1[:,:,:,8:-8]
                S_0 = S_0[:,:,:,8:-8]
                S_1 = S_1[:,:,:,8:-8]
                flow = flow[:,:,:,8:-8]
                    
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
            help="Similarity Loss function. (Default: ncc)",
        )
        parser.add_argument(
            "--ncc_win_size",
            type=int,
            default=9,
            help="Window-Size for the NCC loss function (Default: 9)",
        )

        parser.add_argument(
            "--nmi_bin_size",
            type=int,
            default=64,
            help="Bin-Size for the NMI loss function (Default: 64)",
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

        parser.add_argument(
            "--dropout", action="store_true", help="use dropout"
        )
        parser.add_argument(
            "--savedir", type=str, help="Directory to save images in"
        )

        parser.add_argument(
            "--net", type=str, default="voxelmorph", help="voxelmorph or transmorph"
        )

        return parser
