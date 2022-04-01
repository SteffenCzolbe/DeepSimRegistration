import argparse
import pytorch_lightning as pl
from .autoencoder_model import AutoEncoderModel


def main(hparams):
    # load model
    if hparams.load_from_checkpoint:
        model = AutoEncoderModel.load_from_checkpoint(hparams.load_from_checkpoint)
    else:
        model = AutoEncoderModel(hparams)

    # create early stopping
    hparams.early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=300, strict=True, verbose=True, mode="min"
    )

    # add some hints for better experiments tracking
    hparams.task = "autoencoder"

    # trainer
    trainer = pl.Trainer.from_argparse_args(hparams)

    # fit
    trainer.fit(model)


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()

    # add model specific args
    parser = AutoEncoderModel.add_model_specific_args(parser)

    # add PROGRAM level args
    parser.add_argument(
        "--load_from_checkpoint", help="optional model checkpoint to initialize with"
    )

    # add trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    hparams = parser.parse_args()

    main(hparams)
