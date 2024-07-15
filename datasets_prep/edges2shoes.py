"""
Edges2Shoes dataset in torchmanager dataset.

Code modified from https://github.com/xuekt98/BBDM
"""
from enum import Enum
import os
from typing import Any, Callable
import numpy as np, torchvision
from PIL import Image
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2 as transforms
import torch
import torch.utils.data as data

class Edge2ShoesSplit(Enum):
    TRAIN = 'train'
    VAL = 'val'


class Edge2Shoes(data.Dataset):
    """
    Dataset for the edges2shoes dataset.

    - Properties:
        - image_paths: a `list` of `str` containing the paths to the images.
        - image_size: a `tuple` of `int` containing the size of the images.
        - root_dir: a `str` containing the path to the root directory of the dataset.
        - transform: a `torchvision.transforms.Compose` to transform the images and conditions.
    """
    image_paths: list[str]
    image_size: tuple[int, int]
    root_dir: str
    transform: transforms.Compose

    def __init__(self, path: str, batch_size: int, /, img_size, *, device = None, drop_last: bool = False, num_workers = None, repeat: int = 1, shuffle: bool = False, split: Edge2ShoesSplit = Edge2ShoesSplit.TRAIN):
        """
        Constructor

        - Parameters:
            - path: a `str` containing the path to the root directory of the dataset.
            - batch_size: an `int` containing the batch size.
            - img_size: an `int` or a `tuple` of `int` containing the size of the images.
            - device: a `torch.device` to load the data on.
            - drop_last: a `bool` indicating whether to drop the last batch or not.
            - num_workers: an `int` containing the number of workers to use for loading the data.
            - repeat: an `int` containing the number of times to repeat the dataset.
            - shuffle: a `bool` indicating whether to shuffle the data or not.
            - split: an `Edge2ShoesSplit` indicating which split to use.
        """

        device = torch.device('cpu') if device is None else device
        # initialize
        self.image_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.root_dir = os.path.join(path, split)

        # search for images in folder of root_dir
        self.image_paths = [p for p in os.listdir(os.path.join(self.root_dir)) if os.path.isfile(os.path.join(self.root_dir, p))]
        self.image_paths *= repeat

        # TODO
        self.image_paths = self.image_paths[:2000]

        # initialize transforms
        transforms_list: list[Callable[..., tuple[Any, Any]]] = [transforms.RandomHorizontalFlip(p=0)] if split == Edge2ShoesSplit.TRAIN else []
        transforms_list.extend([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        self.transform = transforms.Compose(transforms_list)

    @property
    def unbatched_len(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # load image
        img_file = self.image_paths[index]
        img_path = os.path.join(self.root_dir, img_file)
        combined_image = Image.open(img_path)
        combined_image = np.array(combined_image)
        w = combined_image.shape[1]
        image = Image.fromarray(combined_image[:, int(w / 2):, :])
        condition = Image.fromarray(combined_image[:, :int(w / 2), :])

        # convert to RGB if necessary
        if not image.mode == 'RGB':
            image = image.convert('RGB')

        # convert to RGB if necessary
        if not condition.mode == 'RGB':
            condition = condition.convert('RGB')

        # apply transform
        image, condition = self.transform(image, condition)
        assert isinstance(image, torch.Tensor), 'Image is not a valid `torch.tensor`.'
        assert isinstance(condition, torch.Tensor), 'Condition is not a valid `torch.tensor`.'

        # normalize image
        image = (image - 0.5) * 2.
        image.clamp(-1., 1.)

        # normalize condition
        condition = (condition - 0.5) * 2.
        condition.clamp(-1., 1.)
        return image, condition

    def __len__(self):
        return len(self.image_paths)
    