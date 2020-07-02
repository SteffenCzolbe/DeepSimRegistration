import os
import argparse
import numpy as np
import torch
import torch.nn.functional as nnf
import torchvision
import pytorch_lightning as pl
import torchreg
import torchreg.transforms as transforms
import torchreg.transforms.functional as ttf
import torchreg.nn as tnn
import torchreg.viz as viz
from .dataset import PlantEMDataset
from .segmentation_model import SegmentationModel
from .models.voxelmorph import Voxelmorph
from .loss_metrics import NCC, DeepSim
from data.platelet_em_reduced import helpers as datahelpers
import ipdb

# set dimensionalty for torchreg layers
torchreg.settings.set_ndims(2)


class RegistrationModel(pl.LightningModule):
    """
    We use pytorch lightning to organize our model code
    """

    def __init__(self, hparams):
        """
        Init method instantiates the network
        """
        super().__init__()
        self.hparams = hparams

        # set net
        self.net = Voxelmorph(
            in_channels=1,
            enc_feat=self.hparams.channels,
            dec_feat=self.hparams.channels[::-1],
            bnorm=self.hparams.bnorm,
            dropout=self.hparams.dropout,
        )

        if hparams.loss == 'deepsim':
            try:
                feature_extractor = SegmentationModel.load_from_checkpoint(hparams.deepsim_weights)
                self.deepsim = DeepSim(feature_extractor)
            except FileNotFoundError:
                feature_extractor = SegmentationModel(hparams)
                self.deepsim = DeepSim(feature_extractor)
                print(f"WARNING: weights '{hparams.deepsim_weights}' not found. skipping!")
        elif hparams.loss in ['ncc', 'ncc+supervised']:
            self.ncc = NCC(window=hparams.ncc_win_size)
        else:
            raise ValueError(f'Loss function "{hparams.loss}" not found.')
        self.mse = torch.nn.MSELoss()
        self.diffusion_reg = torchreg.metrics.GradNorm()
        self.dice_overlap = torchreg.metrics.DiceOverlap(classes=list(range(3)))
        self.transformer = tnn.SpatialTransformer()

        self.viz_every_n_epochs = 25
        self.last_viz = -self.viz_every_n_epochs
        self.viz_batch = next(iter(self.val_dataloader()))

    def dataloader_single_sample(self):
        """
        Dataset of a single sample. Used for debugging purposes
        """
        data = PlantEMDataset(
            "./data/platelet_em_reduced/images/24-images.tif",
            "./data/platelet_em_reduced/labels-class/24-class.tif",
            min_slice=12,
            max_slice=14,
            slice_pairs=True,
            slice_pair_max_z_diff=(0,1),
        )
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=self.hparams.batch_size, drop_last=True
        )
        return dataloader

    def train_dataloader(self):
        """
        Implement one or multiple PyTorch DataLoaders for training.
        """
        if self.hparams.single_sample:
            return self.dataloader_single_sample()

        transform = transforms.VectorizedCompose(
            [
                transforms.VectorizedRandomHorizontalFlip(),
                transforms.VectorizedRandomVerticalFlip(),
                transforms.VectorizedRandomAffine(
                    degrees=180, scale=(0.8, 1.2), shear=20, fillcolor=0
                ),
            ]
        )
        data = PlantEMDataset(
            "./data/platelet_em_reduced/images/50-images.tif",
            "./data/platelet_em_reduced/labels-class/50-class.tif",
            slice_pairs=True,
            slice_pair_max_z_diff=(2,2),
            transform=transform,
        )
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=self.hparams.batch_size, drop_last=True
        )
        return dataloader

    def val_dataloader(self):
        """
        Implement one or multiple PyTorch DataLoaders for validation.
        """
        if self.hparams.single_sample:
            return self.dataloader_single_sample()

        data = PlantEMDataset(
            "./data/platelet_em_reduced/images/24-images.tif",
            "./data/platelet_em_reduced/labels-class/24-class.tif",
            min_slice=0,
            max_slice=12,
            slice_pairs=True,
            slice_pair_max_z_diff=(0,1),
        )
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=self.hparams.batch_size, drop_last=True
        )
        return dataloader

    def test_dataloader(self):
        """
        Implement one or multiple PyTorch DataLoaders for testing.
        """
        if self.hparams.single_sample:
            return self.dataloader_single_sample()

        data = PlantEMDataset(
            "./data/platelet_em_reduced/images/24-images.tif",
            "./data/platelet_em_reduced/labels-class/24-class.tif",
            min_slice=12,
            max_slice=24,
            slice_pairs=True,
            slice_pair_max_z_diff=(0,1),
        )
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=self.hparams.batch_size, drop_last=True
        )
        return dataloader

    def configure_optimizers(self):
        """
        configure optimizers, scheduler, etc
        """
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.1, patience=200, verbose=True)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'epoch', 'monitor': 'val_loss', 'reduce_on_plateau':True}}

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
            return self.ncc(I_m, I_1) + 0.2 * self.mse(S_m_onehot, S_1_onehot) # regularization parameter gamma chosen as 0.2
        else:
            raise ValueError(f'loss function "{self.hparams.loss}" unknow.')
    

    def _step(self, batch, batch_idx, save_viz=False):
        """
        unified step function.
        """
        # unpack batch
        (I_0, S_0), (I_1, S_1) = batch

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

    def viz_results(self, I_0, I_m, I_1, S_0, S_m, S_1, flow, fname=None):

        def class_to_rgb(segmap):
            return (
                ttf.volumetric_image_to_tensor(
                    np.array(datahelpers.class_to_rgb(ttf.image_to_numpy(segmap[0])))
                )
                / 255
            )

        print('creating vizualization...')

        # make figure
        fig = viz.Fig(3, 3, f"Epoch {self.current_epoch}", figsize=(6, 6))
        fig.plot_img(0, 0, I_0[0], vmin=0, vmax=1, title="Source")
        fig.plot_img(0, 1, I_m[0], vmin=0, vmax=1, title="Morphed")
        fig.plot_img(0, 2, I_1[0], vmin=0, vmax=1, title="Target")
        fig.plot_img(
            1, 0, class_to_rgb(S_0), title="Source"
        )
        fig.plot_img(
            1, 1, class_to_rgb(S_m), title="Morphed"
        )
        fig.plot_img(
            1, 2, class_to_rgb(S_1), title="Target"
        )
        fig.plot_transform_grid(
            2, 0, flow[0], title="$\Phi$", interval=15, linewidth=0.1
        )
        fig.plot_transform_vec(2, 1, flow[0], title="$\Phi$", interval=15)

        fig.plot_img(
            2, 2, (S_m[0] - S_1[0]).abs(), vmin=0, vmax=1, title="Diff"
        )


        os.makedirs(self.hparams.savedir, exist_ok=True)
        if fname:
            fig.save(os.path.join(self.hparams.savedir, fname),)
        else:
            fig.save(os.path.join(self.hparams.savedir, f"{self.current_epoch}.pdf"),)

    def mean_dicts(self, list_of_dics):
        """
        means measures over minibatches
        """
        ret = {}
        for k in list_of_dics[0].keys():
            ret[k] = torch.stack([d[k] for d in list_of_dics]).mean()
        return ret

    def training_step(self, batch, batch_idx):
        """
        Operates on a single batch of data from the training set. In this step you’d normally generate examples or calculate
        anything of interest such as accuracy.
        
        Needs to return a dictionary with the entry 'loss'
        """
        return self._step(batch, batch_idx)

    def training_epoch_end(self, outputs):
        """
        Called at the end of the train epoch with the outputs of all training steps.
        """
        output = self.mean_dicts(outputs)
        return {
            "loss": output["loss"],
            "log": dict([(f"train/{k}", v) for k, v in output.items()]),
        }

    def validation_step(self, batch, batch_idx):
        """
        Operates on a single batch of data from the validation set. In this step you’d normally generate examples or calculate
        anything of interest such as accuracy.
        
        Needs to return a dictionary with the entry 'loss'
        """
        # data-handling does not differ from training
        return self._step(batch, batch_idx, save_viz=False)

    def validation_epoch_end(self, outputs):
        """
        Called at the end of the train epoch with the outputs of all training steps.
        """
        # visualize output
        if self.current_epoch - self.last_viz >= self.viz_every_n_epochs:
            device = next(self.parameters()).device
            if device.type == 'cpu' or (device.type == 'cuda' and device.index == 0):
                (I_0, S_0), (I_1, S_1) = self.viz_batch
                batch = (I_0.to(device), S_0.to(device)), (I_1.to(device), S_1.to(device))
                self._step(batch, None, save_viz=True)
                self.last_viz = self.current_epoch

        output = self.mean_dicts(outputs)
        return {
            "val_loss": output["loss"],
            "log": dict([(f"val/{k}", v) for k, v in output.items()]),
        }

    def test_step(self, batch, batch_idx):
        """
        Operates on a single batch of data from the validation set. In this
            "intensity_loss_ncc": intensity_loss,
            "segmentation_loss_mse": segmentation_loss, step you’d normally generate examples or calculate
        anything of interest such as accuracy.
        
        Needs to return a dictionary with the entry 'loss'
        """
        # data-handling does not differ from training
        return self._step(batch, batch_idx, save_viz=False)

    def test_epoch_end(self, outputs):
        """
        Called at the end of the train epoch with the outputs of all training steps.
        """
        output = self.mean_dicts(outputs)
        return {
            "test_loss": output["loss"],
            "log": dict([(f"test/{k}", v) for k, v in output.items()]),
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Adds model specific command-line args
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # optimizer args
        parser.add_argument(
            "--lr", type=float, default=1e-4, help="learning rate (default: 0.0001)"
        )
        parser.add_argument("--loss", type=str, default='ncc', help="Similarity Loss function. Options: 'ncc', 'deepsim', 'ncc+supervised' (Default: ncc)")
        parser.add_argument("--ncc_win_size", type=int, default=9, help="Window-Size for the NCC loss function (Default: 9)")
        parser.add_argument(
            "--deepsim_weights", type=str, default='./loss/weights/segmentation/weights.ckpt', help="Path to deep feature model weights. Default: 'loss/weights/segmentation/weights.ckpt"
        )
        parser.add_argument("--batch_size", type=int, default=8, help="batchsize")
        parser.add_argument(
            "--channels", nargs='+', type=int, default=[64, 128, 256], help="U-Net encoder channels. Decoder uses the reverse. Default: [64, 128, 256]"
        )
        parser.add_argument(
            "--lam", type=float, default=0.5, help="Diffusion regularizer strength"
        )
        parser.add_argument(
            "--bnorm", action='store_true', help="use batchnormalization."
        )
        parser.add_argument(
            "--dropout", action='store_true', help="use dropout"
        )
        parser.add_argument(
            "--single_sample", action='store_true', help="DEBUG OPTION: Set to train on a single sample."
        )

        
        parser.add_argument(
            "--savedir", type=str, help="Directory to save images in"
        )
        return parser
