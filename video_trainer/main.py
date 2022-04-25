from typing import Any, Dict, Tuple

import pytorch_lightning as lightning
import torch
import torchmetrics
import torchvision
from mlflow.pytorch import autolog
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from video_trainer.enums import DatasetSplit
from video_trainer.loading.annotation_file import create as create_annotation_file
from video_trainer.loading.dataset import ConvertBCHWtoCBHW, ImgListToTensor, VideoFrameDataset


class R2Plus1DFineTuned(torch.nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.model = torchvision.models.video.r2plus1d_18(pretrained=True)
        self._set_parameter_requires_grad()
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _set_parameter_requires_grad(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False


class Model(lightning.LightningModule):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.model = R2Plus1DFineTuned()
        self.accuracy = torchmetrics.Accuracy()
        self.f1_score = torchmetrics.F1Score()
        self.prec = torchmetrics.Precision()
        self.recall = torchmetrics.Recall()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        _: Any,
    ) -> Dict[str, torchmetrics.Metric]:
        x, y_true = batch
        y_predicted = self.forward(x)
        loss = self.criterion(y_predicted, y_true)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss  # type: ignore

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _: Any) -> None:
        x, y_true = batch
        y_predicted = self.model(x)

        loss = self.criterion(y_predicted, y_true)
        accuracy = self.accuracy(y_predicted, y_true)
        f1_score = self.f1_score(y_predicted, y_true)
        precision = self.prec(y_predicted, y_true)
        recall = self.recall(y_predicted, y_true)

        self.log('precision', precision)
        self.log('recall', recall)
        self.log('f1_score', f1_score)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def optimizer_zero_grad(
        self, epoch: Any, batch_idx: Any, optimizer: Optimizer, optimizer_idx: Any
    ) -> None:
        optimizer.zero_grad(set_to_none=True)


def main() -> None:
    autolog()
    dataset_train, dataset_validation = create_datasets()
    dataloader_train, dataloader_validation = create_dataloaders(dataset_train, dataset_validation)

    model = Model()
    trainer = lightning.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=5,
        precision=16,
    )
    trainer.fit(
        model=model,
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_validation,
    )


def create_datasets() -> Tuple[Dataset, Dataset]:
    _create_annotation_files()
    preprocess_train = transforms.Compose(
        [
            ImgListToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((150, 128)),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((64, 64)),
            ConvertBCHWtoCBHW(),
        ]
    )
    preprocess_validation = transforms.Compose(
        [
            ImgListToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((150, 128)),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
            ),
            transforms.CenterCrop((64, 64)),
            ConvertBCHWtoCBHW(),
        ]
    )
    return (
        VideoFrameDataset(dataset_split=DatasetSplit.TRAIN, transform=preprocess_train),
        VideoFrameDataset(
            dataset_split=DatasetSplit.VALIDATION, test_mode=True, transform=preprocess_validation
        ),
    )


def _create_annotation_files() -> None:
    create_annotation_file(dataset_split=DatasetSplit.TRAIN)
    create_annotation_file(dataset_split=DatasetSplit.VALIDATION)
    create_annotation_file(dataset_split=DatasetSplit.TEST)


def create_dataloaders(
    dataset_train: Dataset, dataset_validation: Dataset
) -> Tuple[DataLoader, DataLoader]:
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    dataloader_validation = DataLoader(
        dataset=dataset_validation,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return dataloader_train, dataloader_validation


if __name__ == '__main__':
    main()
