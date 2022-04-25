from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from video_trainer.classification.training import train
from video_trainer.enums import DatasetSplit
from video_trainer.loading.annotation_file import create as create_annotation_file
from video_trainer.loading.dataset import ConvertBCHWtoCBHW, ImgListToTensor, VideoFrameDataset


def main() -> None:
    dataset_train, dataset_validation = create_datasets()
    dataloader_train, dataloader_validation = create_dataloaders(dataset_train, dataset_validation)

    train(dataloader_train, dataloader_validation)


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
            transforms.RandomCrop((112, 112)),
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
            transforms.CenterCrop((112, 112)),
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
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    dataloader_validation = DataLoader(
        dataset=dataset_validation,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader_train, dataloader_validation


if __name__ == '__main__':
    main()
