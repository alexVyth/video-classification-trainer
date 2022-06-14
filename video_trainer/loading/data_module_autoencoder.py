import multiprocessing
import os
import random
from typing import List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from video_trainer import settings
from video_trainer.data.video_metadata import VideoData
from video_trainer.enums import DatasetSplit, UnscoredDataset
from video_trainer.loading.annotation_file import create as create_annotation_file
from video_trainer.loading.dataset import VideoFrameDataset
from video_trainer.loading.transforms import ConvertBCHWtoCBHW, ImgListToTensor, NoneTransform
from video_trainer.settings import DATASET_PATH, IMAGE_CROP_SIZE, IMAGE_RESIZE_SIZE

random.seed(10)


class DataModuleAutoEncoding(LightningDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        videos = self._create_video_metadata()
        validation_videos = random.sample(videos, int(0.2 * len(videos)))
        for validation_video in validation_videos:
            validation_video.dataset_split = DatasetSplit.VALIDATION
        train_videos = [video for video in videos if video not in validation_videos]

        create_annotation_file(dataset_split=DatasetSplit.TRAIN, videos_metadata=train_videos)
        create_annotation_file(
            dataset_split=DatasetSplit.VALIDATION, videos_metadata=validation_videos
        )

        self.dataset_train = VideoFrameDataset(
            dataset_split=DatasetSplit.TRAIN,
            test_mode=False,
            transform=self._get_transform(is_train=True),
            has_label=False,
        )
        self.dataset_validation = VideoFrameDataset(
            dataset_split=DatasetSplit.VALIDATION,
            test_mode=True,
            transform=self._get_transform(is_train=False),
            has_label=False,
        )

    def _create_video_metadata(self) -> List[VideoData]:
        videos_metadata = []
        for dataset in UnscoredDataset:
            dataset_path = os.path.join(DATASET_PATH, dataset.value, 'frames_rgb')
            for video in os.listdir(dataset_path):
                video_dir = os.path.join(dataset_path, video)
                videos_metadata.append(
                    VideoData(
                        dataset=dataset, name=video, last_video_frame=len(os.listdir(video_dir))
                    )
                )
        return videos_metadata

    def _get_transform(self, is_train: bool = True) -> transforms.Compose:
        return transforms.Compose(
            [
                ImgListToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize(IMAGE_RESIZE_SIZE),
                transforms.Normalize(
                    mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
                ),
                transforms.RandomHorizontalFlip() if is_train else NoneTransform(),
                transforms.RandomCrop(IMAGE_CROP_SIZE)
                if is_train
                else transforms.CenterCrop(IMAGE_CROP_SIZE),
                ConvertBCHWtoCBHW(),
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=settings.BATCH_SIZE,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset_validation,
            batch_size=settings.BATCH_SIZE,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )
