from typing import Any, Dict, Tuple

import pytorch_lightning as lightning
import torch
import torchmetrics
import torchvision
from torch.optim.optimizer import Optimizer

from video_trainer.settings import (
    BATCH_SIZE,
    EPOCHS,
    FPS,
    FRAMES_PER_SEGMENT,
    IMAGE_RESIZE_SIZE,
    LEARNING_RATE,
    NUM_SEGMENTS,
    PRECISION,
    SAMPLE_DURATION_IN_FRAMES,
)


class System(lightning.LightningModule):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.model = R2Plus1DFineTuned()
        self.accuracy = torchmetrics.Accuracy()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_start(self) -> None:
        self._save_param('model_name', self.model.__class__.__name__)
        self._save_param('loss_function', self.criterion.__class__.__name__)
        self._save_param('sample_duration', SAMPLE_DURATION_IN_FRAMES)
        self._save_param('num_segments', NUM_SEGMENTS)
        self._save_param('frames_per_segment', FRAMES_PER_SEGMENT)
        self._save_param('fps', FPS)
        self._save_param('image_resize_size', IMAGE_RESIZE_SIZE)
        self._save_param('batch_size', BATCH_SIZE)
        self._save_param('epochs', EPOCHS)
        self._save_param('precision', PRECISION)
        self._save_param('learning_rate', LEARNING_RATE)

    def _save_param(self, param_key: str, param_value: Any) -> None:
        self.logger.experiment.log_param(self.logger.run_id, param_key, param_value)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        _: Any,
    ) -> Dict[str, torchmetrics.Metric]:
        x, y_true = batch
        y_predicted = self.forward(x)
        loss = self.criterion(y_predicted, y_true)
        accuracy = self.accuracy(y_predicted, y_true)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss  # type: ignore

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _: Any) -> None:
        x, y_true = batch
        y_predicted = self.model(x)

        loss = self.criterion(y_predicted, y_true)
        accuracy = self.accuracy(y_predicted, y_true)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _: Any) -> None:
        x, y_true = batch
        y_predicted = self.model(x)

        loss = self.criterion(y_predicted, y_true)
        accuracy = self.accuracy(y_predicted, y_true)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE)
        return optimizer

    def optimizer_zero_grad(
        self, epoch: Any, batch_idx: Any, optimizer: Optimizer, optimizer_idx: Any
    ) -> None:
        optimizer.zero_grad(set_to_none=True)


class R2Plus1DFineTuned(torch.nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.name = 'r2plus1d_18'
        self.model = torchvision.models.video.r2plus1d_18(pretrained=True)
        self._set_parameter_requires_grad()
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _set_parameter_requires_grad(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
