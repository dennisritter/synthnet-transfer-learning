"""Implements custom ImageFolder dataset returning image paths in the __getitem__ method."""

from torchvision.datasets import ImageFolder


class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        path = self.imgs[index][0]

        return (img, label, path)
