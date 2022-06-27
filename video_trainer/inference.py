import multiprocessing
from typing import Tuple
import os
import pytorch_lightning as lightning
import torch
import numpy

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

CHECKPOINT_PATH = os.path.join(".","checkpoints","epoch=17-step=95778.ckpt")
seconds_of_pred = 60
frames_per_second = 25
frames_per_sample_all = 32
end = (seconds_of_pred*frames_per_second)//frames_per_sample_all

def main() -> None:
    list_of_samples = [x for x in range(0, end)]
    dataset_train, dataset_validation, dataset_test = create_datasets()
    dataloader_train, dataloader_validation, dataloader_test = create_dataloaders(
        dataset_train, dataset_validation, dataset_test
    )
    model = System.load_from_checkpoint(CHECKPOINT_PATH)
 
    model.eval()
    print("Successfully loaded model!")
    predict_set = torch.utils.data.Subset(dataset_test,list_of_samples)
    trainloader_1 = torch.utils.data.DataLoader(predict_set, batch_size=1,
                                            shuffle=False, num_workers=2)
    print("Successfully made subset!")
    '''
    
    with torch.no_grad():
        pred = model(images)
        print(pred.size())
        print(pred)
    '''
    
    results = []
    for i,batch in enumerate(trainloader_1):
        image,label = batch
        with torch.no_grad():
            y = model(image)
            _,prediction = torch.max(y, dim=1)
        
        y_np = prediction.numpy()
        label_np = label.numpy()
        
        final = numpy.concatenate((y_np,label_np),dtype=str)
        results.append(final)
        
    
    numpy.savetxt('results.txt', results, fmt="%s")    
    

    

    


    

    

    


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
        batch_size=1,
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
