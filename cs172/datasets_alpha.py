import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import string
import torch


from torchvision import transforms
import random


class RandomGrayscaleOrColor:

    def __call__(self, img):
        if random.random() < 0.5:
            return transforms.functional.rgb_to_grayscale(img, num_output_channels=3)
        else:
            return img


class ImageDataset_alpha(Dataset):
    """
    Custom dataset for alphabetic CAPTCHAs (A–Z + a–z), each containing 5 letters.
    The label is one-hot encoded as shape (5, 52).
    """

    def __init__(self, imgdir, transform=None):
        """
        Args:
            imgdir (str): Directory containing CAPTCHA images.
            transform (callable, optional): Transform to apply to images.
        """
        self.imgdir = imgdir
        self.transform = transform
        self.imgs = os.listdir(imgdir)
        self.resize = transforms.Resize((227, 227))

        self.charset = string.ascii_uppercase + string.ascii_lowercase
        self.num_classes = len(self.charset)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.imgdir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.resize(img)

        label_stem = os.path.splitext(self.imgs[idx])[0]
        chars = label_stem.split("#")[0]

        if len(chars) != 5:
            raise ValueError(f"Invalid label format: {self.imgs[idx]}")

        label = torch.zeros((5, self.num_classes), dtype=torch.float32)
        for pos, ch in enumerate(chars):
            if ch not in self.charset:
                raise ValueError(f"Invalid character '{ch}' in label {self.imgs[idx]}")
            label[pos, self.charset.index(ch)] = 1.0

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1))
            img /= 255.0

        return img, label

    def get_samples(self, idx=0):
        """Return both the raw PIL image and its tensor + label"""
        assert idx < self.__len__()
        img_path = os.path.join(self.imgdir, self.imgs[idx])
        image = Image.open(img_path).convert("RGB")
        img, label = self.__getitem__(idx)
        return image, img, label
