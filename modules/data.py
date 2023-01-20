"""Collection of classes and functions related to Classification data."""

import torch
from datasets import load_dataset


class UnNormalize:
    """Undo normalize transform."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def train_val_test_imagefolder(train_dir: str, val_dir: str, test_dir: str):
    """Simple helper to load train, val, test dataset as Huggingface Dataset.

        load datasets from image directory (huggingface)
        https://huggingface.co/docs/datasets/image_dataset
        - Ensure format ds_dir/label/filename_SPLIT.png
        - Each filename has to include the split name
          (e.g.: myname_test, train_my_name, my_val_name)

    Args:
        train_dir (str): Path to train dataset
        val_dir (str): Path to validation dataset
        test_dir (str): Path to test dataset

    Returns:
        Tuple<Dataset>: train, val, test huggingface "imagefolder" datasets
    """
    train_ds = load_dataset("imagefolder", data_dir=train_dir, split="train")
    test_ds = load_dataset("imagefolder", data_dir=test_dir, split="test")
    # Either use given val dataset or else split up test-set into validation and test
    if val_dir:
        val_ds = load_dataset("imagefolder", data_dir=val_dir, split="validation")
    else:
        splits = test_ds.train_test_split(test_size=0.1, stratify_by_column="label")  # stratify
        val_ds = splits["train"]
        test_ds = splits["test"]

    return train_ds, val_ds, test_ds


def collate_fn(examples):
    """Data Collator for Huggingface image classification Dataset.

    Args:
        examples (dict): A batch of data samples

    Returns:
        dict: prepared batch of data samples. Stacked pixel values and tensor of labels
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
