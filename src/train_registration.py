import argparse
import pytorch_lightning as pl
from .registration_model import RegistrationModel


def main(hparams):
    # load model
    if hparams.load_from_checkpoint:
        model = RegistrationModel.load_from_checkpoint(hparams.load_from_checkpoint)
    else:
        model = RegistrationModel(hparams)

    # create early stopping
    hparams.early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=100, strict=True, verbose=True, mode="min"
    )

    # if 
    if hasattr(hparams, 'gpus') and hparams.gpus in [1, '1', [1]]:
        print('Setting to use GPU 0.')
        hparams.gpus = [0]

    # trainer
    trainer = pl.Trainer.from_argparse_args(hparams)

    # fit
    trainer.fit(model)


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()

    # add model specific args
    parser = RegistrationModel.add_model_specific_args(parser)

    # add PROGRAM level args
    parser.add_argument(
        "--load_from_checkpoint", help="optional model checkpoint to initialize with"
    )

    # add trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    hparams = parser.parse_args()

    main(hparams)
