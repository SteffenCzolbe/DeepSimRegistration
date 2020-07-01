import torchreg
import torchreg.transforms as transforms
from .data_access import Image
from torchreg.types import AnnotatedImage, ImageTuple
import numpy as np
import os




def load_img(file_path):
    with np.load(
        os.path.join(file_path)
    ) as data:
        img = data["img"]
        mask = data["mask"] if "mask" in data else None
        segmentation = data["segmentation"] if "segmentation" in data else None
        landmarks = data["landmarks"] if "landmarks" in data else None
        affine = data["affine"] if "affine" in data else None

    return AnnotatedImage(
        img,
        mask,
        segmentation,
        landmarks,
        context={"affine": affine},
    )

def save_img(file_path, annotated_img):
    img = Image().set_image_array_normalized(annotated_img.intensity, annotated_img.mask, annotated_img.context["affine"])
    if annotated_img.segmentation is not None:
        img.add_segmentation_numpy(annotated_img.segmentation)
    if annotated_img.landmarks is not None:
        img.add_landmarks_voxcoord(annotated_img.landmarks)
    img.save(file_path)
    img.save_as_npz(file_path)


if __name__ == '__main__':
    """
    run with
    python3 -m torchreg.transforms.transform_tests.diffeomorphic_transform_test
    """
    torchreg.settings.set_ndims(3)

    # load images
    img0 = load_img('./torchreg/transforms/transform_tests/pre-transform/landmarks/img.npz')
    img1 = load_img('./torchreg/transforms/transform_tests/pre-transform/segmentation/img.npz')

    # transform to tensors
    to_tensor = transforms.ToTensor()
    img0 = to_tensor.transform_annotated_image(img0)
    img1 = to_tensor.transform_annotated_image(img1)

    # form batch
    batch = AnnotatedImage.collate([img0, img1])

    # augment
    augment = transforms.RandomDiffeomorphic(p=1, m=2.5, r=16)
    batch = augment.transform_annotated_image(batch)

    # split batch
    img0, img1 = AnnotatedImage.split(batch)

    # transform back to numpy
    to_numpy = transforms.ToNumpy()
    img0 = to_numpy.transform_annotated_image(img0)
    img1 = to_numpy.transform_annotated_image(img1)

    # save augmented images
    os.makedirs('./torchreg/transforms/transform_tests/post-transform/', exist_ok=True)
    save_img('./torchreg/transforms/transform_tests/post-transform/landmarks/', img0)
    save_img('./torchreg/transforms/transform_tests/post-transform/segmentation/', img1)