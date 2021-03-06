import multiprocessing
from typing import Tuple

import pytorch_lightning as lightning
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from video_trainer import settings
from video_trainer.enums import DatasetSplit
from video_trainer.loading.annotation_file import create as create_annotation_file
from video_trainer.loading.dataset import ConvertBCHWtoCBHW, ImgListToTensor, VideoFrameDataset
from video_trainer.model import System


def main() -> None:

    dataset_train, dataset_validation, dataset_test = create_datasets()
    dataloader_train, dataloader_validation, dataloader_test = create_dataloaders(
        dataset_train, dataset_validation, dataset_test
    )

    trainer = lightning.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=settings.EPOCHS,
        precision=16,
        logger=MLFlowLogger(experiment_name='Default'),
        callbacks=[ModelCheckpoint(save_top_k=3, monitor='val_acc')],
    )

    trainer.fit(
        model=System(),
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_validation,
    )

    trainer.test(ckpt_path='best', dataloaders=dataloader_test)


def create_datasets() -> Tuple[Dataset, Dataset, Dataset]:
    _create_annotation_files()
    transform_train = transforms.Compose(
        [
            ImgListToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(settings.IMAGE_RESIZE_SIZE),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(settings.TRAIN_RANDOM_CROP),
            ConvertBCHWtoCBHW(),
        ]
    )
    transform_validation = transforms.Compose(
        [
            ImgListToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(settings.IMAGE_RESIZE_SIZE),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
            ),
            transforms.CenterCrop(settings.VALID_CENTER_CROP),
            ConvertBCHWtoCBHW(),
        ]
    )
    return (
        VideoFrameDataset(
            dataset_split=DatasetSplit.TRAIN, test_mode=False, transform=transform_train
        ),
        VideoFrameDataset(
            dataset_split=DatasetSplit.VALIDATION, test_mode=True, transform=transform_validation
        ),
        VideoFrameDataset(
            dataset_split=DatasetSplit.TEST, test_mode=True, transform=transform_validation
        ),
    )


def create_dataloaders(
    dataset_train: Dataset,
    dataset_validation: Dataset,
    dataset_test: Dataset,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )
    dataloader_validation = DataLoader(
        dataset=dataset_validation,
        batch_size=4,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )
    return dataloader_train, dataloader_validation, dataloader_test


def _create_annotation_files() -> None:
    create_annotation_file(dataset_split=DatasetSplit.TRAIN)
    create_annotation_file(dataset_split=DatasetSplit.VALIDATION)
    create_annotation_file(dataset_split=DatasetSplit.TEST)


if __name__ == '__main__':
    main()
