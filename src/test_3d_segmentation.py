"""
segments a TIFF image stack
"""
import argparse
import pytorch_lightning as pl
from .segmentation_model import SegmentationModel
import torch
import torchreg.transforms.functional as f
import os
import numpy as np

def same_values(a, b):
    a = a.to('cpu')
    b = b.to('cpu')
    assert torch.allclose(a, b), f"{a}, {b}"

def main(hparams):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = SegmentationModel.load_from_checkpoint(
        checkpoint_path=hparams.weights)
    model.eval()
    model = model.to(device)

    # init trainer
    trainer = pl.Trainer()

    # test (pass in the model)
    #trainer.test(model)

    # segment tiff image stack
    test_set = model.test_dataloader().dataset
    for i in range(10):
        x, y_true = test_set[i]
        x, y_true = x.to(device), y_true.to(device)
        with torch.no_grad():
            y_pred, _ = model.forward(x.unsqueeze(0))
        y_pred = y_pred[0]

        print(f'accuracy of {i}: ', torch.mean((y_true == y_pred).float()))

    os.makedirs(os.path.dirname(hparams.out), exist_ok=True)
    affine = np.array([[  -1.,    0.,    0.,   80.],
                        [   0.,    0.,    1., -112.],
                        [   0.,   -1.,    0.,   96.],
                        [   0.,    0.,    0.,    1.]])
    f.save_tensor_as_nii(os.path.join(hparams.out, 'image.nii.gz'), x, affine=affine, dtype=np.float32)
    f.save_tensor_as_nii(os.path.join(hparams.out, 'gt.nii.gz'), y_true, affine=affine, dtype=np.uint8)
    f.save_tensor_as_nii(os.path.join(hparams.out, 'seg.nii.gz'), y_pred, affine=affine, dtype=np.uint8)

if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument(
        "--weights", type=str,default='./weights/brain-mri/segmentation/weights.ckpt',  help="model checkpoint to initialize with"
    )
    parser.add_argument(
        "--out", type=str, default='./out/brain-mri/segmentation/', help="path to save the segmentation in"
    )

    hparams = parser.parse_args()
    main(hparams)