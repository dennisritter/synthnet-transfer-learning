"""The main run script."""

import lightning as pl

from data.baseline_finetune_dm import BaselineFinetuneDM
from data.generic_finetune_dm import GenericFinetuneDM

pl.seed_everything(0)
dm = GenericFinetuneDM(
    train_dir="data/visda2017/train",
    val_dir="data/visda2017/val",
    test_dir="data/visda2017/test",
)
