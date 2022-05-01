from typing import Any, Dict, List, Tuple

import pytorch_lightning as lightning
import torch
import torchmetrics
import torchvision
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from torch.optim.optimizer import Optimizer

from video_trainer.settings import (
    BATCH_SIZE,
    EPOCHS,
    FPS,
    FRAMES_PER_SEGMENT,
    IMAGE_RESIZE_SIZE,
    LEARNING_RATE,
    NUM_SEGMENTS,
    SAMPLE_DURATION_IN_FRAMES,
)


class System(lightning.LightningModule):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.model = R2Plus1DFineTuned()
        self.accuracy = torchmetrics.Accuracy()
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=5)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_start(self) -> None:
        self._log_param('model_name', self.model.__class__.__name__)
        self._log_param('loss_function', self.criterion.__class__.__name__)
        self._log_param('sample_duration', SAMPLE_DURATION_IN_FRAMES)
        self._log_param('num_segments', NUM_SEGMENTS)
        self._log_param('frames_per_segment', FRAMES_PER_SEGMENT)
        self._log_param('fps', FPS)
        self._log_param('image_resize_size', IMAGE_RESIZE_SIZE)
        self._log_param('batch_size', BATCH_SIZE)
        self._log_param('epochs', EPOCHS)
        self._log_param('learning_rate', LEARNING_RATE)

    def _log_param(self, key: str, value: Any) -> None:
        self.logger.experiment.log_param(self.logger.run_id, key, value)

    def _log_metric(self, key: str, value: Any) -> None:
        self.logger.experiment.log_metric(self.logger.run_id, key, value)

    def _log_image(self, image: Image.Image, directory: str) -> None:
        self.logger.experiment.log_image(self.logger.run_id, image, directory)

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

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], _: Any
    ) -> Dict['str', Any]:
        x, y_true = batch
        y_predicted = self.model(x)

        loss = self.criterion(y_predicted, y_true)
        accuracy = self.accuracy(y_predicted, y_true)
        confusion_matrix = self.confusion_matrix(y_predicted, y_true)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return {'confusion_matrix': confusion_matrix}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _: Any) -> Dict['str', Any]:
        x, y_true = batch
        y_predicted = self.model(x)

        loss = self.criterion(y_predicted, y_true)
        accuracy = self.accuracy(y_predicted, y_true)
        confusion_matrix = self.confusion_matrix(y_predicted, y_true)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return {'confusion_matrix': confusion_matrix}

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        confusion_matrices = [batch['confusion_matrix'] for batch in outputs]
        confusion_matrix = torch.stack(confusion_matrices).sum(dim=0).cpu().numpy()
        confusion_matrix_image = self._get_confusion_matrix_image(confusion_matrix)

        self._log_image(confusion_matrix_image, f'val_confusion_matrix-{self.current_epoch}.png')

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        confusion_matrices = [batch['confusion_matrix'] for batch in outputs]
        confusion_matrix = torch.stack(confusion_matrices).sum(dim=0).cpu().numpy()
        confusion_matrix_image = self._get_confusion_matrix_image(confusion_matrix)

        self._log_image(confusion_matrix_image, 'test_confusion_matrix.png')

    @staticmethod
    def _get_confusion_matrix_image(confusion_matrix: Any) -> Image.Image:
        confusion_matrix_display = ConfusionMatrixDisplay(
            confusion_matrix,
            display_labels=['Climbing', 'Swimming', 'Immobility', 'Diving', 'Head Shake'],
        )
        confusion_matrix_display.plot()
        figure = confusion_matrix_display.figure_
        figure.canvas.draw()
        return Image.frombytes(
            'RGB', figure.canvas.get_width_height(), figure.canvas.tostring_rgb()
        )

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
