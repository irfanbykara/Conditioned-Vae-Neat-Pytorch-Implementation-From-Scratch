from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from typing import Callable, Optional, Tuple

class MNISTAttributes(Dataset):
    """Class utility to load, pre-process, put in batch, and convert to PyTorch convention images from the MNIST dataset.
    """

    def __init__(self, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 mode="train") -> None:
        """ Class initialization.

        :param transform: A set of transformations to apply on data.
        :param target_transform: A set of transformations to apply on labels.
        """
        self.mode = mode
        self.mnist_dataset = MNIST(root='.', train=(mode == "train"),
                                   download=True, transform=transform, target_transform=target_transform)
        self.transform = transform

    def __len__(self):
        """Dataset size.
        :return: Size of the dataset.
        """
        return len(self.mnist_dataset)

    def __getitem__(self, index: int) -> Tuple[ToTensor, int]:
        img, label = self.mnist_dataset[index]

        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     label = self.target_transform(label)

        return img, label


#