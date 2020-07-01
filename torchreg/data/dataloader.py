import torch.utils.data as tdata
import torchreg


class ImageTupleDataloader(tdata.DataLoader):
    def __init__(self, dataset, **kwargs):
        """
        A custom dataloader for image registration of optionally annotated images.   
        Differs from the default pytorch DataLoader in it's way of batching samples.
        
        Parameters:
            dataset: The dataset, eg. IntraPatientRegistrationDataset
            kwargs: key-word arguments for torch.utils.data.Dataloader
        """
        super().__init__(
            dataset, collate_fn=torchreg.types.ImageTuple.collate, **kwargs
        )