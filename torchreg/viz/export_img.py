import torchreg.transforms.functional as transforms
from PIL import Image
import numpy as np


def export_img_2d(path, img):
    """
    exports a tensor as an image. Tensor is comnverted and rescaled from 0..1 to 0..255
    Parameters:
        path: the path to save the image at. eg: './img.png'
        img: a Tensor, CxHxW
    """
    # convert to numpy
    img = transforms.image_to_numpy(img)

    # scale up
    im = Image.fromarray(np.uint8(img * 255))
    im.save(path)
