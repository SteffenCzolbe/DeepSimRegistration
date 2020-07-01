NDIMS = None


def set_ndims(ndims):
    """
    Sets the number of data dimensions. 
    """
    global NDIMS
    NDIMS = ndims


def get_ndims():
    """
    Gets the number of data dimensions. 
    """
    global NDIMS
    if NDIMS is None:
        raise Exception(
            """NDIMS not set. Specify the dimensionality with 'torchreg.settings.set_ndims(ndims)'"""
        )
    return NDIMS
