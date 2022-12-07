from random import random
from typing import Any
from uuid import uuid4

import av
import numpy
import pytorch_lightning as lightning
import torch
import torchmetrics
from torch.nn.functional import mse_loss
from torch.optim.optimizer import Optimizer

from video_trainer.settings import (
    BATCH_SIZE,
    EPOCHS,
    FPS,
    FRAMES_PER_SEGMENT,
    IMAGE_CROP_SIZE,
    IMAGE_RESIZE_SIZE,
    LEARNING_RATE,
    NUM_SEGMENTS,
    SAMPLE_DURATION_IN_FRAMES,
)

VIDEO_DIR = './videos_autoencoder'


class Autoencoder(lightning.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=5)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        y = self.decoder(z)
        return y

    def on_train_start(self) -> None:
        self._log_param('encoder_name', self.encoder.__class__.__name__)
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

    def _log_param(self, key: str, value: Any) -> None:
        self.logger.experiment.log_param(self.logger.run_id, key, value)

    def _log_metric(self, key: str, value: Any) -> None:
        self.logger.experiment.log_metric(self.logger.run_id, key, value)

    def _get_reconstruction_loss(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = mse_loss(x, x_hat, reduction='none')
        loss = loss.sum(dim=[1, 2, 3, 4]).mean(dim=[0])
        return loss

    def training_step(
        self,
        x: torch.Tensor,
        _: Any,
    ) -> torch.Tensor:
        x_hat = self.forward(x)
        loss = mse_loss(x, x_hat, reduction='none')
        loss = loss.sum(dim=[1, 2, 3, 4]).mean(dim=[0])

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, x: torch.Tensor, epoch: int) -> None:
        x_hat = self.forward(x)
        loss = mse_loss(x, x_hat, reduction='none')
        loss = loss.sum(dim=[1, 2, 3, 4]).mean(dim=[0])

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        if random() > 0.99:
            uuid = str(uuid4())
            self._save_video(x, filename=f'{VIDEO_DIR}/{epoch}-{uuid}-input')
            self._save_video(x_hat, filename=f'{VIDEO_DIR}/{epoch}-{uuid}-output')

    def validation_epoch_end(self, _: Any) -> None:
        self.logger.experiment.log_artifacts(run_id=self.logger.run_id, local_dir=VIDEO_DIR)

    def _save_video(self, video_tensor: torch.Tensor, filename: str) -> None:
        total_frames = NUM_SEGMENTS * FRAMES_PER_SEGMENT
        container = av.open(f'{filename}.mp4', mode='w')
        stream = container.add_stream('mpeg4', rate=FPS)
        stream.width = IMAGE_CROP_SIZE[0]
        stream.height = IMAGE_CROP_SIZE[1]
        stream.pix_fmt = 'yuv420p'
        video_numpy = video_tensor.cpu().numpy()[0]

        for i in range(total_frames):
            frame_numpy = video_numpy[:, i, :, :]
            frame_numpy = numpy.round(255 * frame_numpy).astype(numpy.uint8)
            frame_numpy = numpy.swapaxes(frame_numpy, 0, 2)
            frame_numpy = numpy.swapaxes(frame_numpy, 0, 1)
            frame = av.VideoFrame.from_ndarray(frame_numpy, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        container.close()

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer

    def optimizer_zero_grad(self, epoch: Any, _: Any, optimizer: Optimizer, __: Any) -> None:
        optimizer.zero_grad(set_to_none=True)
