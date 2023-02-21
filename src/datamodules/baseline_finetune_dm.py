""" DataModule interpreted from
    A Broad Study of Pre-training for Domain Generalization and Adaptation
    DOI:10.48550/arXiv.2203.11819
    https://www.semanticscholar.org/paper/A-Broad-Study-of-Pre-training-for-Domain-and-Kim-Wang/e0bffb70cd8b5b5ecdc74e1f730dd7298ecc787b
    https://github.com/VisionLearningGroup/Benchmark_Domain_Transfer

    @InProceedings{kim2022unified,
        title={A Broad Study of Pre-training for Domain Generalization and Adaptation},
        author={Kim, Donghyun and Wang, Kaihong and Sclaroff, Stan and Saenko, Kate},
        booktitle = {The European Conference on Computer Vision (ECCV)},
        year = {2022}
    }

    We took info about augmentations, dataset split, and other hardcoded parameters either from the paper
    or (if not specified) from their code (e.g. batch_size, seed, transforms params)
"""

from torchvision import transforms
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split


class BaselineFinetuneDM(pl.LightningDataModule):

    def __init__(
        self,
        train_dir: str,
        test_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
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

    def train_dataloader(self):
        return DataLoader(dataset=self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test, batch_size=self.batch_size, num_workers=self.num_workers)
