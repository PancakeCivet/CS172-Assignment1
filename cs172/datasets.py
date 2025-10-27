"""
This module contains a custom dataset class for loading images and labels from a directory in the CS172 assignment 1.
The ImageDataset class is a subclass of the torch.utils.data.Dataset class, which is a PyTorch class for creating custom datasets.
This class doesn't need to store the images and labels in memory, it just needs to store the path list of the images.
This class is passed to the DataLoader class to load the images and labels in batches.
"""

import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    """
    A custom dataset class for loading images and labels.
    """

    def __init__(self, imgdir, transform=None):
        """
        Args:
            imgdir (str): The directory where the images are stored, the labels are the name of the image.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.imgdir = imgdir
        self.transform = transform
        self.imgs = os.listdir(imgdir)
        self.resize = transforms.Resize((227, 227))

    def __len__(self):
        # ====================== TO DO START ========================
        # Return the number of images in the dataset
        # ===========================================================
        return len(self.imgs)
        # ====================== TO DO END ==========================

    def __getitem__(self, idx):
        # Load the image and label at the given index
        img_path = os.path.join(self.imgdir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.resize(img)

        # ====================== TO DO START ========================
        # return img and label, which should be a np.array or torch.tensor
        # label should be in one-hot format, the shape should be (5, 10)
        # Remeber Pillow Image shape is (H, W, C) and torch need (C, H, W)
        # Remeber Pillow Image type is uint8 ans torch need float32
        # Apply the transform on img if it exists
        # label (from list `self.imgs`) is like '12345#2.png'
        # ===========================================================
        import torch

        label_stem = os.path.splitext(self.imgs[idx])[0]
        digits = label_stem.split("#")[0]

        if len(digits) != 5 or not digits.isdigit():
            raise ValueError(f"Invalid label format: {self.imgs[idx]}")

        label = torch.zeros((5, 10), dtype=torch.float32)
        for position, ch in enumerate(digits):
            label[position, int(ch)] = 1.0

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(
                np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
            )
            img /= 255.0
        # ====================== TO DO END ==========================

        return img, label
    
    def get_samples(self, idx=0):
        assert idx < self.__len__()

        img_path = os.path.join(self.imgdir, self.imgs[idx])
        image = Image.open(img_path).convert("RGB")
        img, label = self.__getitem__(idx)

        return image, img, label
