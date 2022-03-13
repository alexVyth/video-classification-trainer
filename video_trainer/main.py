from typing import Tuple

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from video_trainer.classification.training import train
from video_trainer.enums import DatasetSplit
from video_trainer.loading.annotation_file import create as create_annotation_file
from video_trainer.loading.dataset import ConvertBCHWtoCBHW, ImgListToTensor, VideoFrameDataset


def main() -> None:
    create_annotation_files()

    dataset_train, dataset_validation = create_datasets()
    dataloader_train, dataloader_validation = create_dataloaders(dataset_train, dataset_validation)

    train(dataloader_train, dataloader_validation)


def create_datasets() -> Tuple[Dataset, Dataset]:
    preprocess = transforms.Compose(
        [
            ImgListToTensor(),
            transforms.Resize(32),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
            ),
            ConvertBCHWtoCBHW(),
        ]
    )
    return (
        VideoFrameDataset(dataset_split=DatasetSplit.TRAIN, transform=preprocess),
        VideoFrameDataset(
            dataset_split=DatasetSplit.VALIDATION, test_mode=True, transform=preprocess
        ),
    )


def create_dataloaders(
    dataset_train: Dataset, dataset_validation: Dataset
) -> Tuple[DataLoader, DataLoader]:
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    dataloader_validation = DataLoader(
        dataset=dataset_validation,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader_train, dataloader_validation


def create_annotation_files() -> None:
    create_annotation_file(dataset_split=DatasetSplit.TRAIN)
    create_annotation_file(dataset_split=DatasetSplit.VALIDATION)
    create_annotation_file(dataset_split=DatasetSplit.TEST)


if __name__ == '__main__':
    main()
