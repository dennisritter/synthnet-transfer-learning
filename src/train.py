"""The main run script."""

import os
from typing import List, Optional, Tuple

import hydra
import pyrootutils
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from rich import print

import utils
from datamodules.baseline_finetune_dm import BaselineFinetuneDM
from datamodules.generic_finetune_dm import GenericFinetuneDM

# pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# log = utils.logging_utils.get_logger("rich")
log = utils.get_pylogger(__name__)

# pl.seed_everything(0)
# dm = GenericFinetuneDM(
#     train_dir="data/visda2017/train",
#     val_dir="data/visda2017/val",
#     test_dir="data/visda2017/test",
# )


@utils.task_wrapper
def train(cfg: DictConfig):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)


# python src/train.py --config-dir configs --config-name train.yaml
@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
