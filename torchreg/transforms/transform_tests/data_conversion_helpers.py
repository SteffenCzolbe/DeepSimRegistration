import SimpleITK as sitk
import nibabel as nib
import numpy as np


def world_to_vox(affine, pts):
    """
    transforms the world coordinates given in pts to voxel coordinates.
    :param affine: 4x4 affine transformation matrix, maps homogenous pixel coordinates to world coordinates
    :param pints: Nx3 coordinate array of world-coordinates
    """
    affine_inverse = np.linalg.inv(affine)
    return nib.affines.apply_affine(affine_inverse, pts)


def vox_to_world(affine, pts):
    """
    transforms the world coordinates given in pts to voxel coordinates.
    :param affine: 4x4 affine transformation matrix, maps homogenous pixel coordinates to world coordinates
    :param pints: Nx3 coordinate array of voxel-coordinates
    """
    return nib.affines.apply_affine(affine, pts)


def is_diagonal(M):
    i, j = np.nonzero(M)
    return np.all(i == j)


def build_nii_data(array, affine, header=None):
    """
    Writes the given array as a nifty1 file. 
    :param array: 3D array, containing the image data
    :param affine: 4x4 affine transformation matrix, mapping the voxels of array the a RAS world coordinate system
    :param header: Optional. Nii header. If not provided, the function will create a rudementary one.
    :return: nib.nifti1
    """

    # make a local copy to modify
    affine = np.copy(affine)

    # The array data may be flipped, indicated by negative diagonal values in the affine transformation.
    # This might cause in issue later, when we use more low-level access to the data (eg: deep learning)
    # here we flip it to RAS orientation, and adjust the affine transformation accordingly.
    for ax in range(3):  # iterate over axis
        # search for inverted axis in the affine array
        if affine[ax, ax] < 0:
            # assert that no rotation is involved
            assert is_diagonal(
                affine[:3, :3]
            ), "affine transformation {} is containing rotation elements!".format(
                affine
            )
            # flip axis
            array = nib.orientations.flip_axis(array, axis=ax)
            # flip affine matrix
            affine[ax, ax] *= -1
            # translate affine matrix
            translation = affine[ax, ax] * array.shape[ax]
            affine[ax, -1] -= translation

    # pack the nifty file
    new_nifty = nib.nifti1.Nifti1Image(array.astype(np.int16), affine, header=header)
    new_nifty.update_header()
    return new_nifty


def normalize(array):
    """
    Maps the hounsfield-unit scaled array to 0..1 values.
    A second return contains a mask: 0 if undefined, 1 if defined.
    Mapping:
    intensity:
    < -1024 mapped to 0
    linear scaling
    > 1024 mapped to 1
    
    mask:
    -2000 mapped to 0
    >-2000 mapped to 1 
    """
    shape = array.shape
    mask = np.where(array == -2000, np.zeros(shape), np.ones(shape))
    values = np.maximum(np.minimum(array, 1024), -1024)
    values = (values + 1024) / 2048
    return values, mask


def de_normalize(values, mask):
    """
    Maps the normalized intensity value image and mask back to hounsfield-units.
    Mapping:
    Mask is 0: -2000
    Mask is 1:
        values 0 mapped to -1024
        linear scaling
        values 1 mapped to 1024
    """
    # re-scale values
    values = (values - 0.5) * 2048

    # discretize mask
    mask = mask.round()

    # return spliced result
    # we need a fairly large floating-point tolerance here. IDK why.
    return np.where(mask == 1.0, values, -2000)


class SlicerMarkupBuilder:
    """
    A class to build the markup format for 3D slicer
    """

    def __init__(self):
        self.header = """# Markups fiducial file version = 4.10
# CoordinateSystem = 0
# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID
"""
        self.markups = []

    def append_markup(self, x, y, z):
        """
            appends a markup, in world-coordinates, RAS-coordinate system (x: right, y: anterior/front, z:supra/up)
        """
        self.markups.append((x, y, z))

    def save(self, path):
        if not path.endswith(".fcsv"):
            path += ".fcsv"

        with open(path, "w") as f:
            # append header
            f.write(self.header)

            # append nodes
            for i, (x, y, z) in enumerate(self.markups):
                f.write(
                    f"vtkMRMLMarkupsFiducialNode_{i},{x},{y},{z},0,0,0,1,1,1,0,{i+1},,\n"
                )
