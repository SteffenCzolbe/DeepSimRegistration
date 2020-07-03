"""
segments a TIFF image stack
"""
import argparse
import os
import pytorch_lightning as pl
from .registration_model import RegistrationModel
from tqdm import tqdm
import torchreg.viz as viz


def main(hparams):
    # load model
    model = RegistrationModel.load_from_checkpoint(
        checkpoint_path=hparams.weights)
    model.eval()

    # init trainer
    trainer = pl.Trainer()

    # test (pass in the model)
    #trainer.test(model)

    # create grid animation
    test_set = model.test_dataloader().dataset
    images = []
    for i in tqdm(range(len(test_set)), desc='creating tif image'):
        (I_0, S_0), (I_1, S_1) = test_set[i]
        (I_0, S_0), (I_1, S_1) = (I_0.unsqueeze(0), S_0.unsqueeze(0)), (I_1.unsqueeze(0), S_1.unsqueeze(0))
        flow = model.forward(I_0, I_1)

        fig = viz.Fig(1, 1, None, figsize=(3, 3))
        fig.plot_img(0, 0, I_0[0], vmin=0, vmax=1)
        fig.plot_transform_vec(0, 0, -flow[0], interval=5, arrow_length=1.0, linewidth=1.0, overlay=True)
        # extract the axis we are interested in
        img = fig.save_ax_to_PIL(0, 0)
        images.append(img)
    os.makedirs(os.path.dirname(hparams.out), exist_ok=True)
    images[0].save(hparams.out, save_all=True, append_images=images[1:])


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument(
        "--weights", type=str, default='./weights/phc-u373/registration/ncc+supervised/weights.ckpt', help="model checkpoint to initialize with"
    )
    parser.add_argument(
        "--out", type=str, default='./out/phc-u373/registration/ncc+supervised.tif', help="path to save the result in"
    )

    hparams = parser.parse_args()
    main(hparams)