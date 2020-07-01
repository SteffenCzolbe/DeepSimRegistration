"""
Datatypes to hold information of annotated images for deep learning within the pytorch exosystem.
"""
import torch
from torch.nn.parallel._functions import Scatter
import numpy as np


class ImageTuple(list):
    """
    class to hold multiple images, or a batch of multiple images
    """
    @property
    def source(self):
        return self[0]

    @property
    def target(self):
        return self[1]

    @property
    def morphed(self):
        return self[2]

    def __repr__(self):
        return f'ImageTuple({super().__repr__()})'

    def to(self, device):
        """
        Maps the samples to a device.
        """
        for image in self:
            image.to(device)

        return self

    @staticmethod
    def collate(batch):
        """
        Collates a batch of Image pairs
        """
        # disentagle batches of images
        image_batches = zip(*batch)

        # collate each image separately
        images = [
            type(image_batch[0]).collate(image_batch) for image_batch in image_batches
        ]
        return ImageTuple(images)

    @staticmethod
    def split(batch):
        """
        inverse of collate operation
        """
        # split each image separately
        images = [
            type(image_batch[0]).split(image_batch) for image_batch in batch
        ]

        return [ImageTuple(image_tuple) for image_tuple in zip(*images)]


class AnnotatedImage():
    def __init__(
        self, intensity, mask=None, segmentation=None, landmarks=None, context=None
    ):
        """
        A class to hold the data of one, or a batch of, annotated image for deep learning
        
        Parameters:
            intensity: the image intensity values. Mandatory.
            mask: a mask, 1 if image is defined, 0 if not. Optional. If not provided, all values are assumed to be defined.
            segmentation: 1-hot encoded segmentation data. Optional.
            landmarks: list of landmark annotations. Optional.
            context: additional context information. Optional.
        """
        if intensity is None:
            raise Exception("intensity data is mandatory")
        self.intensity = intensity
        if mask is None:
            mask = np.ones(intensity.shape, dtype=np.float32)
        self.mask = mask
        self.segmentation = segmentation
        self.landmarks = landmarks
        self.context = context

    def items(self):
        return [self.intensity, self.mask, self.segmentation, self.landmarks, self.context]

    def __len__(self):
        return len(self.intensity)

    def to(self, device):
        """
        Maps the sample to a device.
        """

        def obj_to_device(obj, device):
            # generic mapping function
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, list):
                return [obj_to_device(o, device) for o in obj]
            elif obj is None:
                return None
            else:
                raise Exception(f'cannot cast type "{type(obj)}" to device "{device}".')

        # map information usable during deep learning.
        self.intensity = obj_to_device(self.intensity, device)
        self.mask = obj_to_device(self.mask, device)
        self.segmentation = obj_to_device(self.segmentation, device)
        self.landmarks = obj_to_device(self.landmarks, device)
        # Do not map context

        return self

    @staticmethod
    def collate(batch):
        """
        Static method to collate (combine) a batch of one or more lists of samples.
        
        The default collate_fn can not handle optional data types, or tensors of differing size. Thus we implement our own.
        """
        # batch of single AnnotatedImage
        intensity = torch.stack([b.intensity for b in batch])
        mask = torch.stack([b.mask for b in batch])
        segmentation = TensorList([b.segmentation for b in batch])
        landmarks = TensorList([b.landmarks for b in batch])
        context = TensorList([b.context for b in batch])
        return AnnotatedImage(intensity, mask, segmentation, landmarks, context)

    @staticmethod
    def split(batch):
        """
        Static method to split (combine) a batch of one or more lists of samples.

        inverse of collate
        """
        # batch of single AnnotatedImage
        intensity = [row for row in batch.intensity]
        mask =  [row for row in batch.mask]
        segmentation = batch.segmentation
        landmarks = batch.landmarks
        context = batch.context
        return [AnnotatedImage(*args) for args in zip(intensity, mask, segmentation, landmarks, context)]

class TensorList(list): 
    """
    A class for batching Tensors of different size.
    """
    @staticmethod
    def from_list(l):
        if l is None:
            return None
        if isinstance(l, TensorList):
            return l
        return TensorList(l)

    def __repr__(self):
        return f'TensorList({super().__repr__()})'


"""
We overwrite the scatter function to distribute Annotated images across many GPUs
"""
def scatter_modif(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """

    def scatter_map(obj):
        # extended version of the default scatter_map function
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        if isinstance(obj, AnnotatedImage):
            return scatter_annotated_image(obj)
        return [obj for targets in target_gpus]

    def scatter_annotated_image(annotated_image):
        # scatter intensity and masks values first
        intensities = scatter_map(annotated_image.intensity)
        mask = scatter_map(annotated_image.mask)

        # do some acrobatics to scatter remaining lists accordingly
        # WARNING: this breaks backpropagation of the TensorList entries when going across devices
        devices = [intensity.device for intensity in intensities]
        num_per_device = [len(intensity) for intensity in intensities]
        segmentation = scatter_tensor_list(
            annotated_image.segmentation, devices, num_per_device
        )
        landmarks = scatter_tensor_list(annotated_image.landmarks, devices, num_per_device)
        context = scatter_tensor_list(annotated_image.context, devices, num_per_device)
        return list(
            map(
                lambda x: AnnotatedImage(*x),
                zip(intensities, mask, segmentation, landmarks, context),
            )
        )

    def scatter_tensor_list(lst, devices, num_per_device):
        # scatters a list of tensors across devices
        # WARNING: this breaks backpropagation of the TensorList entries when going across devices
        res = []
        j = 0
        for device_no, device in enumerate(devices):
            r = []
            for i in range(num_per_device[device_no]):
                nxt = lst[j]
                if isinstance(nxt, torch.Tensor):
                    nxt = nxt.to(device)
                r.append(nxt)
                j += 1
            res.append(r)
        return TensorList(res)

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None
    return res

# overwrite library function
import torch.nn.parallel.scatter_gather
torch.nn.parallel.scatter_gather.scatter = scatter_modif