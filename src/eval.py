import pickle  # nosec
from typing import List, Tuple

import faiss
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

import utils

# pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # assert cfg.ckpt_path
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, num_classes=datamodule.num_classes)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # Make sure you check enabled transforms for train_loader (you probably want to disable them)
    train_loader = datamodule.train_dataloader(shuffle=False)
    test_loader = datamodule.test_dataloader()

    log.info("Run [TRAIN DATASET] Predictions!")
    predictions = trainer.predict(model=model, dataloaders=train_loader, ckpt_path=cfg.ckpt_path)
    # print(predictions)

    train_preds, train_targets, train_logits, train_features, train_paths = [], [], [], [], []
    for pd in predictions:
        train_preds += pd["preds"].tolist()
        train_targets += pd["targets"].tolist()
        train_logits += pd["logits"].tolist()
        train_features += pd["features"]
        train_paths += pd["paths"]
    train_features = [feat.numpy() for feat in train_features]
    train_predictions = {
        "preds": train_preds,
        "targets": train_targets,
        "logits": train_logits,
        "features": train_features,
        "paths": train_paths,
    }

    log.info("Run [TEST DATASET] Predictions!")
    predictions = trainer.predict(model=model, dataloaders=test_loader, ckpt_path=cfg.ckpt_path)

    test_preds, test_targets, test_logits, test_features, test_paths = [], [], [], [], []
    for pd in predictions:
        test_preds += pd["preds"].tolist()
        test_targets += pd["targets"].tolist()
        test_logits += pd["logits"].tolist()
        test_features += pd["features"]
        test_paths += pd["paths"]
    test_features = [feat.numpy() for feat in test_features]
    test_predictions = {
        "preds": test_preds,
        "targets": test_targets,
        "logits": test_logits,
        "features": test_features,
        "paths": test_paths,
    }

    path_feature_train = {
        path: feature for path, feature in zip(train_predictions["paths"], train_predictions["features"])
    }
    path_feature_test = {
        path: feature for path, feature in zip(test_predictions["paths"], test_predictions["features"])
    }
    with open(f"{cfg.paths.output_dir}/path_feature_train.pkl", "wb") as f:
        pickle.dump(path_feature_train, f)
    with open(f"{cfg.paths.output_dir}/path_feature_test.pkl", "wb") as f:
        pickle.dump(path_feature_test, f)

    feature_size = train_predictions["features"][0].shape[0]
    index_flat_l2 = faiss.IndexFlatL2(feature_size)
    index_flat_l2.add(np.array(train_predictions["features"]))  # pylint: disable=no-value-for-parameter
    faiss.write_index(index_flat_l2, f"{cfg.paths.output_dir}/index_flat_l2.faiss")

    # metric_dict = trainer.callback_metrics
    # return metric_dict, object_dict


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
