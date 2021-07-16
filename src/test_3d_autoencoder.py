"""
segments a TIFF image stack
"""
import argparse
import pytorch_lightning as pl
from .autoencoder_model import AutoEncoderModel
import torch
import torchreg.transforms.functional as f
import os
import numpy as np


def main(hparams):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = AutoEncoderModel.load_from_checkpoint(checkpoint_path=hparams.weights)
    model.eval()
    model = model.to(device)

    print(f"Evaluating model for dataset {model.hparams.dataset}")

    # init trainer
    trainer = pl.Trainer()

    # test (pass in the model)
    # trainer.test(model)

    # segment tiff image stack
    test_set = model.test_dataloader().dataset
    for i in range(min(10, len(test_set))):
        x, _ = test_set[i]
        x = x.to(device)
        with torch.no_grad():
            x_pred= model.forward(x.unsqueeze(0))
        x_pred = x_pred[0]
        print(x_pred[0,50,50,:])

        print(f"mean_Absolute distance of {i}: ", torch.abs((x - x_pred).mean()))

        os.makedirs(os.path.dirname(hparams.out), exist_ok=True)
        affine = np.array(
            [
                [-1.0, 0.0, 0.0, 80.0],
                [0.0, 0.0, 1.0, -112.0],
                [0.0, -1.0, 0.0, 96.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        f.save_tensor_as_nii(
            os.path.join(hparams.out, f"{i}image.nii.gz"),
            x,
            affine=affine,
            dtype=np.float32,
        )
        f.save_tensor_as_nii(
            os.path.join(hparams.out, f"{i}recon.nii.gz"),
            x_pred,
            affine=affine,
            dtype=np.float32,
        )


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument(
        "--weights",
        type=str,
        default="./weights/brain-mri/autoencoder/weights.ckpt",
        help="model checkpoint to initialize with",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./out/brain-mri/autoencoder/",
        help="path to save the reconstruction in",
    )

    hparams = parser.parse_args()
    main(hparams)
