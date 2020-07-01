import torchvision.transforms as tvt
import torchvision.transforms.functional as F
import random
import PIL

"""
torchvision transforms for image tuples.
"""


class VectorizedToTensor(tvt.ToTensor):
    def __init__(self):
        super().__init__()

    def __call__(self, *imgs):
        return [super(VectorizedToTensor, self).__call__(img) for img in imgs]


class VectorizedCompose(tvt.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, *imgs):
        for t in self.transforms:
            imgs = t(*imgs)
        return imgs


class VectorizedRandomAffine(tvt.RandomAffine):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image And int for grayscale) for the area
            outside the transform in the output image.(Pillow>=5.0.0)

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *imgs):
        """
            imgs (Iterable of PIL Image): Images to be transformed.

        Returns:
            PIL Images: Iterable of P Affine transformed image.
        """
        # get random transformation params
        param = self.get_params(
            self.degrees, self.translate, self.scale, self.shear, imgs[0].size
        )
        # apply transformation
        return [
            F.affine(img, *param, resample=self.resample, fillcolor=self.fillcolor)
            for img in imgs
        ]


class VectorizedRandomHorizontalFlip(tvt.RandomHorizontalFlip):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__(p=p)

    def __call__(self, *imgs):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return [F.hflip(img) for img in imgs]
        return imgs


class VectorizedRandomVerticalFlip(tvt.RandomVerticalFlip):
    """vertical flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__(p=p)

    def __call__(self, *imgs):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return [F.vflip(img) for img in imgs]
        return imgs

class VectorizedColorJitter(tvt.ColorJitter):
    def __init__(self, *args, **kwargs):
        """Randomly change the brightness, contrast and saturation of an image.

        Args:
            brightness (float or tuple of float (min, max)): How much to jitter brightness.
                brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
                or the given [min, max]. Should be non negative numbers.
            contrast (float or tuple of float (min, max)): How much to jitter contrast.
                contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
                or the given [min, max]. Should be non negative numbers.
            saturation (float or tuple of float (min, max)): How much to jitter saturation.
                saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
                or the given [min, max]. Should be non negative numbers.
            hue (float or tuple of float (min, max)): How much to jitter hue.
                hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
                Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        """
        super().__init__(*args, **kwargs)

    def __call__(self, *imgs):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return [enhance_contrast(img,2) for img in imgs]