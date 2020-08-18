"""
segments a TIFF image stack
"""
import argparse
import os
import torch
import torchreg
import torchreg.transforms.functional as f
import pytorch_lightning as pl
from .registration_model import RegistrationModel
from tqdm import tqdm
import torchreg.viz as viz
import numpy as np


def main(hparams):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = RegistrationModel.load_from_checkpoint(
        checkpoint_path=hparams.weights)
    model.eval()
    model = model.to(device)

    transformer = torchreg.nn.SpatialTransformer()

    print(f'Evaluating model for dataset {model.hparams.dataset}, loss {model.hparams.loss}, lambda {model.hparams.lam}')

    # init trainer
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0)

    # test (pass in the model)
    trainer.test(model)

    # create grid animation
    test_set = model.test_dataloader().dataset
    for i in tqdm(range(1), desc='registering..'):
        (I_0, S_0), (I_1, S_1) = test_set[i]
        (I_0, S_0), (I_1, S_1) = (I_0.unsqueeze(0).to(device), S_0.unsqueeze(0).to(device)), (I_1.unsqueeze(0).to(device), S_1.unsqueeze(0).to(device))
        flow = model.forward(I_0, I_1)
        I_m = transformer(I_0, flow)
        S_m = transformer(S_0.float(), flow, mode='nearest').round().long()

        print(f'accuracy of {i}: ', torch.mean((S_m == S_1).float()))

        os.makedirs(os.path.dirname(hparams.out), exist_ok=True)
        affine = np.array([[  -1.,    0.,    0.,   80.],
                            [   0.,    0.,    1., -112.],
                            [   0.,   -1.,    0.,   96.],
                            [   0.,    0.,    0.,    1.]])
        f.save_tensor_as_nii(os.path.join(hparams.out, f'{i}i_0.nii.gz'), I_0[0], affine=affine, dtype=np.float32)
        f.save_tensor_as_nii(os.path.join(hparams.out, f'{i}s_0.nii.gz'), S_0[0], affine=affine, dtype=np.uint8)
        f.save_tensor_as_nii(os.path.join(hparams.out, f'{i}i_1.nii.gz'), I_1[0], affine=affine, dtype=np.float32)
        f.save_tensor_as_nii(os.path.join(hparams.out, f'{i}s_1.nii.gz'), S_1[0], affine=affine, dtype=np.uint8)
        f.save_tensor_as_nii(os.path.join(hparams.out, f'{i}i_m.nii.gz'), I_m[0], affine=affine, dtype=np.float32)
        f.save_tensor_as_nii(os.path.join(hparams.out, f'{i}s_m.nii.gz'), S_m[0], affine=affine, dtype=np.uint8)
        f.save_tensor_as_np(os.path.join(hparams.out, f'{i}flow.np'), flow[0])


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument(
        "--weights", type=str, default='./weights/brain-mri/registration/ncc+supervised/weights.ckpt', help="model checkpoint to initialize with"
    )
    parser.add_argument(
        "--out", type=str, default='./out/brain-mri/registration/ncc+supervised/', help="path to save the result in"
    )

    hparams = parser.parse_args()
    main(hparams)