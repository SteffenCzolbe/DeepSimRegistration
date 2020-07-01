"""
segments a TIFF image stack
"""
import argparse
import os
import pytorch_lightning as pl
import torchreg.transforms.functional as ttf
from .registration_model import RegistrationModel
from .dataset import PlantEMDataset
from data.platelet_em_reduced import helpers as datahelpers
from tqdm import tqdm


def main(hparams):
    # create out directory
    os.makedirs(hparams.out, exist_ok=True)

    # load model
    model = RegistrationModel.load_from_checkpoint(
        checkpoint_path=hparams.weights)
    model.eval()

    # init trainer
    trainer = pl.Trainer()

    # test single image
    test_set = model.test_dataloader().dataset
    (a,b), (c,d) = test_set[4]
    batch = (a.unsqueeze(0), b.unsqueeze(0)), (c.unsqueeze(0), d.unsqueeze(0))
    model._step(batch, None)

    # test (pass in the model)
    trainer.test(model)

    


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument(
        "--weights", type=str,default='./v2/weights/registration/lam_0.5/weights.ckpt',  help="model checkpoint to initialize with"
    )
    parser.add_argument(
        "--out", type=str, default='./v2/out/registration/test', help="path to save the result in"
    )

    hparams = parser.parse_args()
    main(hparams)