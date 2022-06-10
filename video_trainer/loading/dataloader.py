import multiprocessing
from typing import Tuple

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from video_trainer import settings


def create_dataloaders(
    dataset_train: Dataset,
    dataset_validation: Dataset,
) -> Tuple[DataLoader, DataLoader]:
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )
    dataloader_validation = DataLoader(
        dataset=dataset_validation,
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )
    return dataloader_train, dataloader_validation
