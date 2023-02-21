"""The main run script."""

from typing import List, Optional, Tuple

import hydra
import pyrootutils
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from rich import print

from datamodules.baseline_finetune_dm import BaselineFinetuneDM
from datamodules.generic_finetune_dm import GenericFinetuneDM
from utils import logging_utils

# pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = logging_utils.get_logger("rich")

# pl.seed_everything(0)
# dm = GenericFinetuneDM(
#     train_dir="data/visda2017/train",
#     val_dir="data/visda2017/val",
#     test_dir="data/visda2017/test",
# )


def train(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg), extra={"markup": True})
    if cfg.get("seed"):
        log.info(cfg.seed)
        pl.seed_everything(cfg.seed, workers=True)


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
