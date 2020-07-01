"""
Dataset-independent representation of intra-patient samples
"""
import numpy as np
import pickle
import os
import nibabel as nib
import torchreg.transforms.transform_tests.data_conversion_helpers as datahelpers
import json


class Image:
    """
    A sample representing a single scan
    """

    def __init__(self):
        self.nii = None
        self.landmarks = None
        self.segmentation_nii = None

    def add_image_nifty(self, nii):
        """
        :param nii: nifty image, RAS+ coordinate system
        """
        self.nii = nii
        return self

    def add_image_numpy(self, array, affine, header=None):
        """
        :param array: 3d volumetric image, RAS+ coordinate system
        :param affine: 4x4 affine transformation matrix, maps homogenous pixel coordinates to world coordinates
        """
        self.nii = datahelpers.build_nii_data(array, np.copy(affine), header=header)
        return self

    def add_segmentation_nifty(self, segmentation_nii):
        """
        :param segmentation_nii: optional, segmentation mask as an Nii image. Lung = 1, Not lung = 0
        """
        self.segmentation_nii = segmentation_nii
        return self

    def add_segmentation_numpy(self, segmentation, affine=None):
        """
        :param segmentation: segmentation mask as an numpy array image. Lung = 1, Not lung = 0
        """
        if affine is None:
            affine = self.get_affine()
        self.segmentation_nii = datahelpers.build_nii_data(
            segmentation, affine, header=self.get_header()
        )
        return self

    def add_landmarks_voxcoord(self, landmarks):
        landmarks_world = datahelpers.vox_to_world(self.get_affine(), landmarks)
        self.add_landmarks_worldcoord(landmarks_world)
        return self

    def add_landmarks_worldcoord(self, landmarks):
        self.landmarks = landmarks
        return self

    def get_img_array(self):
        """
        returns the image array in houndfield units
        """
        return self.nii.get_fdata(caching="unchanged", dtype=np.float32)

    def get_img_array_normalized(self):
        """
        returns the image and a mask
        The image encodes the normalized intensity values (0..1)
        The mask encodes if the image is defined at this place. 1: is defined, 0: not defined.
        """
        return datahelpers.normalize(self.get_img_array())

    def set_img_array(self, img, affine=None):
        """
        sets the image array, in hounsfield units
        """
        if affine is None:
            affine = self.get_affine()
        new_nifty = nib.nifti1.Nifti1Image(
            img.astype(np.int16), affine, header=self.get_header()
        )
        new_nifty.update_header()
        self.nii = new_nifty
        return

    def set_image_array_normalized(self, img, mask, affine=None):
        """
        sets the image array, in normalized units as returned bu the get_img_array_normalized function.
        """
        if affine is None:
            affine = self.get_affine()
        self.set_img_array(datahelpers.de_normalize(img, mask), affine)
        return self

    def get_header(self):
        if self.nii:
            return self.nii.header
        else:
            return None

    def get_affine(self):
        return np.copy(self.nii.affine)

    def set_affine(self, affine):
        # hack to set affine array
        self.nii.affine[:] = affine
        # update the nii file header
        self.nii.update_header()
        if self.has_segmentation():
            # do the same for segmentation data
            self.segmentation_nii.affine[:] = affine
            self.segmentation_nii.update_header()
        return

    def get_resolution(self):
        """
        returns the resolution (in mm)
        """
        return self.nii.get_header().get_zooms()

    def get_shape(self):
        """
        returns the shape of the array (voxel count)
        """
        return self.nii.get_header().get_data_shape()

    def has_landmarks(self):
        return self.landmarks is not None and self.landmarks.size > 0

    def get_landmarks_voxcoord(self):
        if self.has_landmarks():
            return datahelpers.world_to_vox(self.get_affine(), self.landmarks)
        else:
            return np.array([[]])

    def get_landmarks_worldcoord(self):
        if self.has_landmarks():
            return self.landmarks
        else:
            return np.array([[]])

    def has_segmentation(self):
        return self.segmentation_nii is not None

    def get_segmentation_array(self):
        return self.segmentation_nii.get_fdata(
            caching="unchanged", dtype=np.float16
        ).astype(
            np.uint8
        )  # ugly data type conversions, but this libary doesnt allow ints

    def set_segmentation_array(self, seg_array):
        new_nifty = nib.nifti1.Nifti1Image(
            seg_array.astype(np.uint8), self.get_affine(), header=self.get_header()
        )
        new_nifty.update_header()
        self.segmentation_nii = new_nifty
        return

    @staticmethod
    def load(path):
        """
        Loads a sample from file
        :param path: save location
        :type path: str
        :return: the loaded Image
        :rtype:
        """
        image = Image()

        # read image
        nii = nib.load(os.path.join(path, "img.nii.gz"))
        image.add_image_nifty(nii)

        # read segmentation
        if os.path.isfile(os.path.join(path, "seg.nii.gz")):
            segmentation_nii = nib.load(os.path.join(path, "seg.nii.gz"))
            image.add_segmentation_nifty(segmentation_nii)

        # read landmarks
        if os.path.isfile(os.path.join(path, "landmarks.pts")):
            landmarks = np.loadtxt(os.path.join(path, "landmarks.pts"))
            image.add_landmarks_worldcoord(landmarks)

        return image

    def save(self, path):
        """
        saves the sample to file
        :param path: save location
        :return: None
        """
        os.makedirs(path, exist_ok=True)

        # save image
        nib.save(self.nii, os.path.join(path, "img.nii.gz"))

        # save segmentation
        if self.has_segmentation():
            nib.save(self.segmentation_nii, os.path.join(path, "seg.nii.gz"))

        if self.has_landmarks():
            # save landmarks
            np.savetxt(os.path.join(path, "landmarks.pts"), self.landmarks)

            # save landmarks in 3D slicer format
            builder = datahelpers.SlicerMarkupBuilder()
            for (x, y, z) in self.get_landmarks_worldcoord():
                builder.append_markup(x, y, z)
            builder.save(os.path.join(path, "markups.fcsv"))
        return

    def save_as_npz(self, path):
        """
        saves anormalized, voxel-coord version as an npz file
        invludes img, segmentation, landmarks, affine
        """
        kwargs = {}
        img, mask = self.get_img_array_normalized()
        kwargs["img"] = img
        kwargs["mask"] = mask
        kwargs["affine"] = self.get_affine()
        if self.has_landmarks():
            kwargs["landmarks"] = self.get_landmarks_voxcoord()
        if self.has_segmentation():
            kwargs["segmentation"] = self.get_segmentation_array()
        np.savez(os.path.join(path, "img.npz"), **kwargs)

    @staticmethod
    def load_from_npz(self, path):
        """
        loads the image from an npz archive
        """
        npz = np.load(os.path.join(path, "img.npz"))
        img = Image().set_image_array_normalized(npz["img"], npz["mask"], npz["affine"])
        if "landmarks" in npz.keys():
            img.add_landmarks_voxcoord(npz["landmarks"])
        if "segmentation" in npz.keys():
            img.add_segmentation_numpy(npz["segmentation"])
        return img
