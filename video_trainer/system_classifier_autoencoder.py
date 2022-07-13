from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy
import pandas
import pytorch_lightning as lightning
import seaborn
import torch
import torchmetrics
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from torch.optim.optimizer import Optimizer

from video_trainer.models import ClassifierV3
from video_trainer.settings import (
    BATCH_SIZE,
    EPOCHS,
    FPS,
    FRAMES_PER_SEGMENT,
    HAS_FROZEN_WEIGHTS,
    IMAGE_CROP_SIZE,
    IMAGE_RESIZE_SIZE,
    LEARNING_RATE,
    NUM_SEGMENTS,
    SAMPLE_DURATION_IN_FRAMES,
)
from video_trainer.system_autoencoder import Autoencoder


class System(lightning.LightningModule):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.model = Autoencoder.load_from_checkpoint(
            './mlruns/0/822b9fd95fde4a85b76e215218322b65/checkpoints/epoch=29-step=83400.ckpt'
        )
        self.model.decoder = ClassifierV3()
        self.accuracy = torchmetrics.Accuracy()
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=5)
        self.f1_score = torchmetrics.F1Score(num_classes=5, average='none')
        self.prec = torchmetrics.Precision(num_classes=5, average='none')
        self.recall = torchmetrics.Recall(num_classes=5, average='none')
        self.criterion = torch.nn.CrossEntropyLoss()

        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded_features = self.model.encoder(x)
        return self.model.decoder(encoded_features)

    def on_train_start(self) -> None:
        self._log_param('model_name', self.model.decoder.__class__.__name__)
        self._log_param('loss_function', self.criterion.__class__.__name__)
        self._log_param('sample_duration', SAMPLE_DURATION_IN_FRAMES)
        self._log_param('num_segments', NUM_SEGMENTS)
        self._log_param('frames_per_segment', FRAMES_PER_SEGMENT)
        self._log_param('fps', FPS)
        self._log_param('image_resize_size', IMAGE_RESIZE_SIZE)
        self._log_param('image_crop_size', IMAGE_CROP_SIZE)
        self._log_param('batch_size', BATCH_SIZE)
        self._log_param('epochs', EPOCHS)
        self._log_param('learning_rate', LEARNING_RATE)
        self._log_param('has_frozen_weights', HAS_FROZEN_WEIGHTS)

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
        y_predicted = self.forward(x)

        loss = self.criterion(y_predicted, y_true)
        accuracy = self.accuracy(y_predicted, y_true)
        confusion_matrix = self.confusion_matrix(y_predicted, y_true)
        precision = self.prec(y_predicted, y_true)
        f1_score = self.f1_score(y_predicted, y_true)
        recall = self.recall(y_predicted, y_true)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return {
            'confusion_matrix': confusion_matrix,
            'precision': precision,
            'f1_score': f1_score,
            'recall': recall,
        }

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _: Any) -> Dict['str', Any]:
        x, y_true = batch
        y_predicted = self.forward(x)

        loss = self.criterion(y_predicted, y_true)
        accuracy = self.accuracy(y_predicted, y_true)
        confusion_matrix = self.confusion_matrix(y_predicted, y_true)
        precision = self.prec(y_predicted, y_true)
        f1_score = self.f1_score(y_predicted, y_true)
        recall = self.recall(y_predicted, y_true)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return {
            'confusion_matrix': confusion_matrix,
            'precision': precision,
            'f1_score': f1_score,
            'recall': recall,
        }

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        confusion_matrices = [output['confusion_matrix'] for output in outputs]
        confusion_matrix = torch.stack(confusion_matrices).sum(dim=0).cpu().numpy()
        confusion_matrix_image = self._get_confusion_matrix_image(confusion_matrix)
        self._log_image(confusion_matrix_image, f'val_confusion_matrix-{self.current_epoch}.png')

        precisions = [output['precision'] for output in outputs]
        precision = torch.stack(precisions).mean(dim=0).cpu().numpy()

        recalls = [output['recall'] for output in outputs]
        recall = torch.stack(recalls).mean(dim=0).cpu().numpy()

        f1_scores = [output['f1_score'] for output in outputs]
        f1_score = torch.stack(f1_scores).mean(dim=0).cpu().numpy()

        classification_report_image = self._get_classification_report_image(
            precision, recall, f1_score
        )
        self._log_image(
            classification_report_image, f'val_classification_report-{self.current_epoch}.png'
        )

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        confusion_matrices = [output['confusion_matrix'] for output in outputs]
        confusion_matrix = torch.stack(confusion_matrices).sum(dim=0).cpu().numpy()
        confusion_matrix_image = self._get_confusion_matrix_image(confusion_matrix)

        self._log_image(confusion_matrix_image, 'test_confusion_matrix.png')

        precisions = [output['precision'] for output in outputs]
        precision = torch.stack(precisions).mean(dim=0).cpu().numpy()

        recalls = [output['recall'] for output in outputs]
        recall = torch.stack(recalls).mean(dim=0).cpu().numpy()

        f1_scores = [output['f1_score'] for output in outputs]
        f1_score = torch.stack(f1_scores).mean(dim=0).cpu().numpy()

        classification_report_image = self._get_classification_report_image(
            precision, recall, f1_score
        )
        self._log_image(
            classification_report_image,
            'test_classification_report.png',
        )

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

    @staticmethod
    def _get_classification_report_image(
        precision: Any, recall: Any, f1_score: Any
    ) -> Image.Image:
        plt.figure().clear()
        labels = ['Climbing', 'Swimming', 'Immobility', 'Diving', 'Head Shake']
        columns = ['precision', 'recall', 'f1 score']
        data = pandas.DataFrame(
            numpy.stack([precision, recall, f1_score]).T, index=labels, columns=columns
        )
        plot = seaborn.heatmap(data, vmin=0, vmax=1, cmap='YlGnBu', annot=True)
        return Image.frombytes(
            'RGB', plot.figure.canvas.get_width_height(), plot.figure.canvas.tostring_rgb()
        )

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer

    def optimizer_zero_grad(
        self, epoch: Any, batch_idx: Any, optimizer: Optimizer, optimizer_idx: Any
    ) -> None:
        optimizer.zero_grad(set_to_none=True)