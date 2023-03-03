"""Callbacks for pytorch lightning trainer."""

import numpy as np
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

import utils
from utils.transforms import UnNormalize

log = utils.get_pylogger(__name__)


class LogPredictionSamplesCallback(Callback):
    def __init__(self, n: int = 4):
        super().__init__()
        self.n = n

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        if batch_idx == 0:
            if self.n > len(batch):
                self.n = len(batch)
            x, y = batch
            images = [img for img in x[: self.n]]
            idx2label = trainer.datamodule.idx2label
            captions = [
                f"gt: {idx2label[y_i.item()]} | pred: {idx2label[pred_i.item()]}"
                for y_i, pred_i in zip(y[: self.n], outputs["preds"][: self.n])
            ]
            trainer.logger.log_image(key="prediction_samples", images=images, caption=captions)


class LogTrainingSamplesCallback(Callback):
    def __init__(self, n: int = 4):
        super().__init__()
        self.n = n

    def on_train_start(self, trainer, pl_module) -> None:
        dm = trainer.datamodule
        # original_images = [dm.train[i][0] for i in range(0, self.n)]
        # labels = [dm.idx2label[dm.train[i][1]] for i in range(0, self.n)]
        loader = DataLoader(dataset=dm.train, batch_size=self.n, num_workers=0, shuffle=True)
        samples = next(iter(loader))
        labels = [dm.idx2label[label_i.item()] for label_i in samples[1]]

        # trainer.logger.log_image(key="original_training_samples", images=original_images, caption=labels)
        trainer.logger.log_image(key="transformed_training_samples", images=list(samples[0]), caption=labels)
        return super().on_train_start(trainer, pl_module)
