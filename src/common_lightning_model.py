import os
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
import torchreg
import torchreg.transforms as transforms
from .datasets.tif_stack_dataset import TiffStackDataset
from .datasets.brain_mri_dataset import BrainMRIDataset


class CommonLightningModel(pl.LightningModule):
    """
    We use pytorch lightning to organize our model code
    """

    def __init__(self, hparams, dataset_image_pairs=False):
        """
        Init method instantiates the network
        """
        super().__init__()
        self.hparams = hparams
        self.image_pairs = dataset_image_pairs

        # set dimensionalty for torchreg layers
        torchreg.settings.set_ndims(self.dataset_config('dim'))

        # set-up data visualization
        self.viz_every_n_epochs = self.hparams.viz_every_n_epochs
        self.last_viz = -self.viz_every_n_epochs
        self.viz_batch = next(iter(self.val_dataloader()))

    def dataset_config(self, key):
        if self.hparams.dataset == 'platelet-em':
            config = {'dataset_type':'tif',
                'channels' : 1,
                'classes': 3,
                'class_colors': [(0, 40, 97), (0, 40, 255), (255, 229, 0)],
                'dim': 2,
                'train_intensity_image_file':"./data/platelet_em_reduced/images/50-images.tif",
                'train_segmentation_image_file':"./data/platelet_em_reduced/labels-class/50-class.tif",
                'train_image_slice_from_to':(0, -1),
                'val_intensity_image_file':"./data/platelet_em_reduced/images/24-images.tif",
                'val_segmentation_image_file':"./data/platelet_em_reduced/labels-class/24-class.tif",
                'val_image_slice_from_to':(0, 12),
                'test_intensity_image_file':"./data/platelet_em_reduced/images/24-images.tif",
                'test_segmentation_image_file':"./data/platelet_em_reduced/labels-class/24-class.tif",
                'test_image_slice_from_to':(12, 24),
                'reduce_lr_patience': 200,}
        elif self.hparams.dataset == 'phc-u373':
            config = {'dataset_type':'tif',
                'channels' : 1,
                'classes': 2,
                'class_colors': [(0, 0, 0), (27, 247, 156)],
                'dim': 2,
                'train_intensity_image_file':"./data/PhC-U373/images/01.tif",
                'train_segmentation_image_file':"./data/PhC-U373/labels-class/01.tif",
                'train_image_slice_from_to':(0, -1),
                'val_intensity_image_file':"./data/PhC-U373/images/02.tif",
                'val_segmentation_image_file':"./data/PhC-U373/labels-class/02.tif",
                'val_image_slice_from_to':(0, 50),
                'test_intensity_image_file':"./data/PhC-U373/images/02.tif",
                'test_segmentation_image_file':"./data/PhC-U373/labels-class/02.tif",
                'test_image_slice_from_to':(60, 115),
                'reduce_lr_patience': 200,}
        elif self.hparams.dataset == 'brain-mri':
            config = {'dataset_type':'nii',
                'channels' : 1,
                'classes': 24,
                'dim': 3,
                'path': '../brain_mris',
                'reduce_lr_patience': 2,}
        else:
            raise ValueError(f'Dataset "{self.hparams.dataset}" not known.')
        return config[key]

    def make_dataloader(self, split='train'):
        """
        Implement one or multiple PyTorch DataLoaders for training.
        """
        torchreg.settings.set_ndims(self.dataset_config('dim'))
        if self.dataset_config('dataset_type') == 'tif':
            self.augmentation = transforms.RandomAffine(degrees=(-180, 180), translate=(-1, 1), scale=(0.8, 1.2), shear=(-0.03, 0.03), flip=True)
            data = TiffStackDataset(
                intensity_tif_image=self.dataset_config(f'{split}_intensity_image_file'),
                segmentation_tif_image=self.dataset_config(f'{split}_segmentation_image_file'),
                min_slice=self.dataset_config(f'{split}_image_slice_from_to')[0],
                max_slice=self.dataset_config(f'{split}_image_slice_from_to')[1],
                slice_pairs=self.image_pairs,
                slice_pair_max_z_diff=(2,2),
            )
        elif self.dataset_config('dataset_type') == 'nii':
            self.augmentation = transforms.RandomAffine(degrees=(-5, 5), translate=None, scale=None, shear=None, flip=True)
            data = BrainMRIDataset(
                path=self.dataset_config('path'),
                split=split,
                pairs=self.image_pairs,
            )

        dataloader = torch.utils.data.DataLoader(
            data, batch_size=self.hparams.batch_size, drop_last=True
        )
        return dataloader

    def train_dataloader(self):
        return self.make_dataloader('train')

    def val_dataloader(self):
        return self.make_dataloader('val')

    def test_dataloader(self):
        return self.make_dataloader('test')

    def configure_optimizers(self):
        """
        configure optimizers, scheduler, etc
        """
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.1, patience=self.dataset_config('reduce_lr_patience'), verbose=True)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'epoch', 'monitor': 'val_loss', 'reduce_on_plateau':True}}

    def forward(self, x):
        """
        Same as torch.nn.Module.forward(), however in Lightning you want this to
        define the operations you want to use for prediction (i.e.: on a server or as a feature extractor).
        """

        raise NotImplementedError()

    def _step(self, batch, batch_idx, save_viz=False):
        """
        unified step function.
        """
        raise NotImplementedError()

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
        return self._step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        """
        Called at the end of the train epoch with the outputs of all training steps.
        """
        def map_to_device(obj, device):
            if isinstance(obj, tuple):
                return tuple(map_to_device(list(obj), device))
            elif isinstance(obj, list):
                return list(map(lambda o: map_to_device(o, device), obj))
            else:
                return obj.to(device)
        # visualize output
        if self.current_epoch - self.last_viz >= self.viz_every_n_epochs:
            device = next(iter(self.parameters())).device
            if device.type == 'cpu' or (device.type == 'cuda' and device.index == 0):
                batch = map_to_device(self.viz_batch, device)
                print('Creating Visualization..')
                import ipdb; ipdb.set_trace()
                self._step(batch, None, save_viz=True)
                self.last_viz = self.current_epoch
        
        output = self.mean_dicts(outputs)
        return {
            "val_loss": output["loss"],
            "log": dict([(f"val/{k}", v) for k, v in output.items()]),
        }

    def test_step(self, batch, batch_idx):
        """
        Operates on a single batch of data from the validation set. In this step you’d normally generate examples or calculate
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
    def add_common_model_args():
        parser = argparse.ArgumentParser(add_help=False)

        parser.add_argument(
            "--dataset", type=str, help="Dataset. Options: 'platelet-em'"
        )
        parser.add_argument(
            "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
        )
        parser.add_argument("--batch_size", type=int, default=8, help="batchsize")
        parser.add_argument(
            "--viz_every_n_epochs", type=int, default=25, help="Visualization every n epochs. Default 25."
        )
        return parser
