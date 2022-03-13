from typing import Tuple

from torch.utils.data.dataset import Dataset

from video_trainer.enums import DatasetSplit
from video_trainer.loading.annotation_file import create as create_annotation_file
from video_trainer.loading.dataset import VideoFrameDataset


def main() -> None:
    create_annotation_files()
    dataset_train, dataset_validation, dataset_test = create_datasets()

    for _ in range(64):
        video, label = next(iter(dataset_train))
        print(video)
        print(label)

    for _ in range(64):
        video, label = next(iter(dataset_validation))
        print(video)
        print(label)

    for _ in range(64):
        video, label = next(iter(dataset_test))
        print(video)
        print(label)


def create_datasets() -> Tuple[Dataset, Dataset, Dataset]:
    return (
        VideoFrameDataset(dataset_split=DatasetSplit.TRAIN),
        VideoFrameDataset(dataset_split=DatasetSplit.VALIDATION, test_mode=True),
        VideoFrameDataset(dataset_split=DatasetSplit.TEST, test_mode=True),
    )


def create_annotation_files() -> None:
    create_annotation_file(dataset_split=DatasetSplit.TRAIN)
    create_annotation_file(dataset_split=DatasetSplit.VALIDATION)
    create_annotation_file(dataset_split=DatasetSplit.TEST)


if __name__ == '__main__':
    main()
