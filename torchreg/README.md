# Torchreg

A toolkit for deep-learning based image registration with pytorch.

Similar to torchvision, but for dimensionality-agnostic (works in 2d AND 3d out of the box), and nonvinient transformations for both images and annotations (segmentation masks and landmarks!).

## Important Packages

- nn - includes common building blocks for deep learning models, such as spatial transformer blocks, flow integration, flow composition, model_io etc.
- transforms - includes transforms to map and augment 2d and 3d voxel-based data.
- metrics - various metrics useful for registration
- data - dataset and dataloader classes

## Guide

We first import torchreg and set up the dimensionality. Many operations infere their working based on the dimensionality of the data. Currently supported are 2d and 3d voxel-based images

```
import torchreg
torchreg.settings.set_dims(3)
```