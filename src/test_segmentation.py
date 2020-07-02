"""
segments a TIFF image stack
"""
import argparse
import pytorch_lightning as pl
from .segmentation_model import SegmentationModel
from tqdm import tqdm
import os


def main(hparams):
    # load model
    model = SegmentationModel.load_from_checkpoint(
        checkpoint_path=hparams.weights)

    # init trainer
    trainer = pl.Trainer()

    # test (pass in the model)
    trainer.test(model)

    # segment tiff image stack
    test_set = model.test_dataloader().dataset
    segmented_images = []
    for i in tqdm(range(len(test_set)), desc='creating tif image'):
        x, y_true = test_set[i]
        x, y_true = x.unsqueeze(0), y_true.unsqueeze(0)
        y_pred, _ = model.forward(x)
        viz = model.viz_results(x, y_true, y_pred, save=False)
        # extract the axis we are interested in
        img = viz.save_ax_to_PIL(0, 1)
        segmented_images.append(img)
    os.makedirs(os.path.dirname(hparams.out), exist_ok=True)
    segmented_images[0].save(hparams.out, save_all=True, append_images=segmented_images[1:])

    


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument(
        "--weights", type=str,default='./weights/phc-u373/segmentation/weights.ckpt',  help="model checkpoint to initialize with"
    )
    parser.add_argument(
        "--out", type=str, default='./out/phc-u373/segmentation/segmentation.tif', help="path to save the segmentation in"
    )

    hparams = parser.parse_args()
    main(hparams)