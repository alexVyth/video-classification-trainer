from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from video_trainer import settings
from video_trainer.data.video_metadata import VIDEO_METADATA
from video_trainer.enums import DatasetSplit
from video_trainer.loading.annotation_file import create as create_annotation_file
from video_trainer.loading.dataset import VideoFrameDataset
from video_trainer.loading.transforms import ConvertBCHWtoCBHW, ImgListToTensor, NoneTransform
from video_trainer.settings import IMAGE_CROP_SIZE, IMAGE_RESIZE_SIZE


class DataModuleClassification(LightningDataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        create_annotation_file(dataset_split=DatasetSplit.TRAIN, videos_metadata=VIDEO_METADATA)
        create_annotation_file(
            dataset_split=DatasetSplit.VALIDATION, videos_metadata=VIDEO_METADATA
        )
        self.dataset_train = VideoFrameDataset(
            dataset_split=DatasetSplit.TRAIN,
            test_mode=False,
            transform=self._get_transform(is_train=True),
        )
        self.dataset_validation = VideoFrameDataset(
            dataset_split=DatasetSplit.VALIDATION,
            test_mode=True,
            transform=self._get_transform(is_train=False),
        )

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
            num_workers=2,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset_validation,
            batch_size=settings.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
