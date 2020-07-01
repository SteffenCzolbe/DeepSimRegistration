"""
methods to combine channels and color masks of 2d images.
"""
import torch


def toRGB(Gray=None, R=None, G=None, B=None):
    """
    Combines up to 4 color channels of format B x 1 x H x W to an RGB image tensor
    """
    # get input shape
    for arg in [Gray, R, G, B]:
        if arg is not None:
            shape = arg.shape
            device = arg.device
            dtype = arg.dtype

    # create output tensor
    img = torch.zeros((shape[0], 3, *shape[2:]), device=device, dtype=dtype)
    # add values to output tensor
    if Gray is not None:
        img[:, 0, ...] = Gray
        img[:, 1, ...] = Gray
        img[:, 2, ...] = Gray
    if R is not None:
        img[:, 0, ...] = R
    if G is not None:
        img[:, 1, ...] = G
    if B is not None:
        img[:, 2, ...] = B
    return img


def interpolate(a, b, alpha):
    """
    interpolates between two images a and b with weight alpha
    returns (1-alpha)*a + alpha*b
    """
    return (1 - alpha) * a + alpha * b


def make_const_color_like(a, R, G, B):
    """
    Makes a constant colored image shape and type as a
    Parameters:
        a: tensor to copy shape and type from
        R, G, B: color code, floats.
    """
    shape = a.shape
    device = a.device
    dtype = a.dtype
    img = torch.zeros((shape[0], 3, *shape[2:]), device=device, dtype=dtype)
    img[:, 0] = R
    img[:, 1] = G
    img[:, 2] = B
    return img


def make_chequered_like(a, width=5):
    """
    Makes a chequered pattern of shape and type as a
    Parameters:
        a: tensor to copy shape and type from
        width: chequer pattern width. Default 5
    """
    # make black background
    cheq = torch.zeros_like(a)

    # add red boxes
    for i in range(width):
        for j in range(width):
            # (odd_rows, even_columns)
            cheq[:, 0, (i + width) :: width * 2, j :: width * 2] = 0.5
            # (even_rows, odd_columns)
            cheq[:, 0, j :: width * 2, (i + width) :: width * 2] = 0.5

    return cheq
