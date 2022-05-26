# from random import sample
import os

from video_trainer.data.video_metadata import VIDEO_METADATA
from video_trainer.enums import DatasetSplit
from video_trainer.settings import TEMP_DIR

VIDEOS = {video.name for video in VIDEO_METADATA}

VALIDATION_VIDEOS = {
    '4C-2012-F15',
    '4C-2012-F19',
    '4C-2012-F23',
    '4C-2012-F32',
    '4C-2012-F41',
    '4C-F4',
    '4C-F7',
    '4C-F8',
    '4C-F18',
    '4C-F29',
    '4C-F30',
    '4C-F31',
    '4C-F37',
    '4C-F39',
    '4C-F45',
    '4C-F46',
    '4C-M5',
    '4C-M7',
    '4C-M15',
    '4C-M21',
}

TRAIN_VIDEOS = VIDEOS - VALIDATION_VIDEOS

DATASET_SPLIT_TO_VIDEOS = {
    DatasetSplit.TRAIN: TRAIN_VIDEOS,
    DatasetSplit.VALIDATION: VALIDATION_VIDEOS,
}

DATASET_SPLIT_TO_ANNOTATION_PATH = {
    DatasetSplit.TRAIN: os.path.join(TEMP_DIR, 'train_sample_data.txt'),
    DatasetSplit.VALIDATION: os.path.join(TEMP_DIR, 'validation_sample_data.txt'),
}
