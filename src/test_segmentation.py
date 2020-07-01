"""
segments a TIFF image stack
"""
import argparse
import pytorch_lightning as pl
from .segmentation_model import SegmentationModel
from tqdm import tqdm


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
        y_pred, _ = model.forward(x.unsqueeze(0))
        img = datahelpers.class_to_rgb(ttf.image_to_numpy(y_pred[0]))
        segmented_images.append(img)
    segmented_images[0].save(hparams.out, save_all=True, append_images=segmented_images[1:])

    


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument(
        "--weights", type=str,default='./v2/weights/segmentation/weights.ckpt',  help="model checkpoint to initialize with"
    )
    parser.add_argument(
        "--out", type=str, default='./v2/out/segmented.tif', help="path to save the segmentation in"
    )

    hparams = parser.parse_args()
    main(hparams)