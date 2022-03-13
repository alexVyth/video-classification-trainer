from typing import Tuple

from torch.utils.data.dataset import Dataset

from video_trainer.loading.annotation_file import create as create_annotation_file
from video_trainer.loading.dataset import VideoFrameDataset
from video_trainer.settings import (
    TEST_ANNOTATION_PATH,
    TRAIN_ANNOTATION_PATH,
    VALIDATION_ANNOTATION_PATH,
)


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
        VideoFrameDataset(annotationfile_path=TRAIN_ANNOTATION_PATH),
        VideoFrameDataset(annotationfile_path=VALIDATION_ANNOTATION_PATH, test_mode=True),
        VideoFrameDataset(annotationfile_path=TEST_ANNOTATION_PATH, test_mode=True),
    )


def create_annotation_files() -> None:
    create_annotation_file(annotation_file=TRAIN_ANNOTATION_PATH)
    create_annotation_file(annotation_file=VALIDATION_ANNOTATION_PATH)
    create_annotation_file(annotation_file=TEST_ANNOTATION_PATH)


if __name__ == '__main__':
    main()
