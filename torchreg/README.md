# Torchreg

A toolkit for deep-learning based image registration with pytorch.

Similar to torchvision, but for dimensionality-agnostic (works in 2d AND 3d out of the box), and nonvinient transformations for both images and annotations (segmentation masks and landmarks!).

## Important Packages

- nn - includes common building blocks for deep learning models, such as spatial transformer blocks, flow integration, flow composition, model_io etc.
- transforms - includes transforms to map and augment 3d voxel-based data.
- types - The basic datatypes nessesary to use torchreg - the AnnotatedImage and ImageTuple
- metrics - various metrics useful for registration
- data - dataset and dataloader classes

## Guide

We first import torchreg and set up the dimensionality. Many operations infere their working based on the dimensionality of the data. Currently supported are 2d and 3d voxel-based images

```
import torchreg
torchreg.settings.set_dims(3)
```

Next, we load some data. torchreg groups images and annotations as an `AnnotatedImage`. The grouping allows us to easily access and process an image together with annotations. An annotated image contains image intensity values, and optionally any of:

- `mask`: defining where the image indensity is defined. A mask is `0` where the image is not defined, and `1` if it is defined
- `segmentation`: a segmentation mask
- `landmarks`: a list of points, annotating certain features in the image
- `context`: additonal imformation that we might want to save with the image, such as file names, disease information, etc...

Typically, we want to register one or more images to each other. For this purpose, we can group any number of images as a `ImageTuple`:

```
from torchreg.types import AnnotatedImage, ImageTuple

# load your information and map it to the AnnotatedImage
source = AnnotatedImage(img, mask, segmentation, landmarks, context)
tagret = AnnotatedImage(img, mask, segmentation, landmarks, context)
registration_pair = ImageTuple(source, target)
```

Any dataset utelizing the torchreg package needs to yield such ImageTuples. To perform common pre-processing on `ImageTuple`s, you can use the transformers in `torchreg.transforms`. For example `torchreg.transforms.ToTensor` maps ImageTuples to torch's tensors.

The `torchreg.data.ImageTupleDataloader` will load your custom dataset, and do all the batching and collating of data for you:

```
dataset = MyCustomDataset(fnames, )
loader = torchreg.data.ImageTupleDataloader(dataset, batch_size)
```

During training, the information can now be easily accessed and evaluated using:

```
# set-up model and loss functions
model = MyModel().to('cuda')
transformer = torchreg.nn.ApplyTransform().to('cuda')
mse = torchreg.metrics.MaskedMSE()
dice = torchreg.metrics.DiceOverlap()

# iterate over batches
for source, target in loader:
    source.to('cuda')
    target.to('cuda')

    flow, inv_flow = model(source.intensity, target.intensity)

    morphed = transformer(source, flow, inv_flow)

    mse_loss = mse(morphed, target)
    dice_loss = dice(morphed, target)
```
