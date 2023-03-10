"""A generic DataModule for fine-tuning."""

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class GenericFinetuneDM(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str = None,
        val_dir: str = None,
        test_dir: str = None,
        batch_size: int = 64,
        num_workers: int = 4,
        toy: bool = False,
        image_size: tuple = (224, 224),
        image_mean: list = [0.485, 0.456, 0.406],
        image_std: list = [0.229, 0.224, 0.225],
        random_horizontal_flip: bool = True,
        random_vertical_flip: bool = False,
        random_color_jitter: bool = False,
        random_grayscale: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.toy = toy
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.random_horizontal_flip = random_horizontal_flip
        self.random_vertical_flip = random_vertical_flip
        self.random_color_jitter = random_color_jitter
        self.random_grayscale = random_grayscale

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size, scale=(0.7, 1.0)),
                transforms.RandomApply([transforms.RandomHorizontalFlip()], p=int(self.random_horizontal_flip)),
                transforms.RandomApply([transforms.RandomVerticalFlip()], p=int(self.random_vertical_flip)),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)],
                    p=int(self.random_color_jitter),
                ),
                transforms.RandomApply([transforms.RandomGrayscale()], p=int(self.random_grayscale)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.val_dir:
            self.train = ImageFolder(self.train_dir)
            self.val = ImageFolder(self.val_dir)
        else:
            self.train, self.val = random_split(ImageFolder(self.train_dir), [0.8, 0.2])
            self.train = self.train.dataset
            self.val = self.val.dataset
        self.test = ImageFolder(self.test_dir)

        self.train.transform = self.train_transform
        self.val.transform = self.val_transform
        self.test.transform = self.val_transform

        self.num_classes = len(self.train.classes)
        self.label2idx = self.train.class_to_idx
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        # If toy is set true, use a very small subset of the data just for testing
        # Use 80 samples for training and 20 for testing
        if self.toy:
            self.train = Subset(self.train, np.random.choice(np.arange(len(self.train)), size=80, replace=False))
            self.val = Subset(self.val, np.random.choice(np.arange(len(self.val)), size=80, replace=False))
            self.test = Subset(self.test, np.random.choice(np.arange(len(self.test)), size=80, replace=False))

    def train_dataloader(self):
        return DataLoader(dataset=self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test, batch_size=self.batch_size, num_workers=self.num_workers)
