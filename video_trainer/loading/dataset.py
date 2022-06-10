import os
import os.path
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from video_trainer.data.splitting import DATASET_SPLIT_TO_ANNOTATION_PATH
from video_trainer.enums import DatasetSplit
from video_trainer.loading.annotation_file import create as create_annotation_file
from video_trainer.settings import (
    DATASET_PATH,
    FRAMES_PER_SEGMENT,
    FRAMES_RGB_TEMPLATE,
    IMAGE_CROP_SIZE,
    IMAGE_RESIZE_SIZE,
    NUM_SEGMENTS,
)


class VideoRecord:
    def __init__(self, row: List[str], root_datapath: str):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])

    @property
    def path(self) -> str:
        return self._path

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def start_frame(self) -> int:
        return int(self._data[1])

    @property
    def end_frame(self) -> int:
        return int(self._data[2])

    @property
    def label(self) -> int:
        return int(self._data[3])


class VideoFrameDataset(Dataset):
    def __init__(
        self,
        dataset_split: DatasetSplit,
        num_segments: int = NUM_SEGMENTS,
        frames_per_segment: int = FRAMES_PER_SEGMENT,
        transform: Any = None,
        test_mode: bool = False,
    ):
        super().__init__()

        self.annotationfile_path = DATASET_SPLIT_TO_ANNOTATION_PATH[dataset_split]
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.transform = transform
        self.test_mode = test_mode

        self._parse_annotationfile()
        self._sanity_check_samples()

    @staticmethod
    def _load_image(directory: str, idx: int) -> Image.Image:
        return Image.open(os.path.join(directory, FRAMES_RGB_TEMPLATE.format(idx))).convert('RGB')

    def _parse_annotationfile(self) -> None:
        self.video_list = []
        with open(self.annotationfile_path, encoding='utf-8') as annotation_file:
            for annotation_file_row in annotation_file:
                row = annotation_file_row.strip().split()
                self.video_list.append(VideoRecord(row, DATASET_PATH))

    def _sanity_check_samples(self) -> None:
        for record in self.video_list:
            if record.num_frames <= 0 or record.start_frame == record.end_frame:
                print(f'video {record.path} seems to have no RGB frames')

            elif record.num_frames < (self.num_segments * self.frames_per_segment):
                print(
                    f'\nDataset Warning: video {record.path} has {record.num_frames} frames '
                    f'error when trying to load this video.\n'
                )

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[List[Image.Image], int], Tuple[torch.Tensor, int], Tuple[Any, int]]:
        record: VideoRecord = self.video_list[idx]

        frame_start_indices: np.ndarray = self._get_start_indices(record)

        return self._get(record, frame_start_indices)

    def _get_start_indices(self, record: VideoRecord) -> np.ndarray:
        if self.test_mode:
            distance_between_indices = (record.num_frames - self.frames_per_segment + 1) / float(
                self.num_segments
            )

            start_indices = np.array(
                [
                    int(distance_between_indices / 2.0 + distance_between_indices * x)
                    for x in range(self.num_segments)
                ]
            )
        else:
            max_valid_start_index = (
                record.num_frames - self.frames_per_segment + 1
            ) // self.num_segments

            start_indices = np.multiply(
                list(range(self.num_segments)), max_valid_start_index
            ) + np.random.randint(max_valid_start_index, size=self.num_segments)

        return start_indices

    def _get(
        self, record: VideoRecord, frame_start_indices: np.ndarray
    ) -> Union[Tuple[List[Image.Image], int], Tuple[torch.Tensor, int], Tuple[Any, int]]:
        frame_start_indices = frame_start_indices + record.start_frame
        images = []
        for start_index in frame_start_indices:
            frame_index = int(start_index)
            for _ in range(self.frames_per_segment):
                image = self._load_image(record.path, frame_index)
                images.append(image)

                if frame_index < record.end_frame:
                    frame_index += 1

        if self.transform is not None:
            images = self.transform(images)

        return images, record.label

    def __len__(self) -> int:
        return len(self.video_list)


class ImgListToTensor(torch.nn.Module):
    @staticmethod
    def forward(
        img_list: List[Image.Image],
    ) -> torch.Tensor:
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])


class ConvertBCHWtoCBHW(torch.nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)"""

    @staticmethod
    def forward(vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)


def _create_annotation_files() -> None:
    create_annotation_file(dataset_split=DatasetSplit.TRAIN)
    create_annotation_file(dataset_split=DatasetSplit.VALIDATION)


def create_datasets() -> Tuple[Dataset, Dataset]:
    _create_annotation_files()
    transform_train = transforms.Compose(
        [
            ImgListToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(IMAGE_RESIZE_SIZE),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(IMAGE_CROP_SIZE),
            ConvertBCHWtoCBHW(),
        ]
    )
    transform_validation = transforms.Compose(
        [
            ImgListToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(IMAGE_RESIZE_SIZE),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
            ),
            transforms.CenterCrop(IMAGE_CROP_SIZE),
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
    )
