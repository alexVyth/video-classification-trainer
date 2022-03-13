# from random import sample
import os

from video_trainer.data.video_metadata import VIDEO_METADATA
from video_trainer.enums import DatasetSplit
from video_trainer.settings import DATASET_PATH

VIDEOS = {video.name for video in VIDEO_METADATA}

# test_videos = sample(videos, round(len(VIDEO_METADATA)* 0.2))
TEST_VIDEOS = {
    '4C-F1',
    '4C-M21',
    '4C-F48',
    '4C-2012-F22',
    '4C-F21',
    '4C-2012-F25',
    '4C-2012-F19',
    '4C-F29',
    '4C-F16',
    '4C-F9',
    '4C-F18',
    '4C-F2',
    '4C-2012-F29',
    '4C-F33',
    '4C-F14',
    '4C-F46',
    '4C-M1',
    '4C-2012-F24',
    '4C-2012-F13',
    '4C-2012-F38',
    '4C-M17',
    '4C-M23',
}

TRAIN_VALIDATION_VIDEOS = VIDEOS - TEST_VIDEOS

# validation_videos = sample(training_validation_videos, round(len(VIDEO_METADATA)* 0.2))
VALIDATION_VIDEOS = {
    '4C-F40',
    '4C-2012-F17',
    '4C-M25',
    '4C-F31',
    '4C-F36',
    '4C-F42',
    '2',
    '4C-F8',
    '4C-2012-F21',
    '4C-F38',
    '4C-2012-F9',
    '4C-F45',
    '4C-2012-F4',
    '4C-M33',
    '3',
    '4C-M37',
    '4C-F15',
    '4C-M27',
    '4C-M19',
    '25',
    '4C-2012-F27',
    '4C-F47',
}
TRAIN_VIDEOS = TRAIN_VALIDATION_VIDEOS - VALIDATION_VIDEOS

DATASET_SPLIT_TO_VIDEOS = {
    DatasetSplit.TRAIN: TRAIN_VIDEOS,
    DatasetSplit.VALIDATION: VALIDATION_VIDEOS,
    DatasetSplit.TEST: TEST_VIDEOS,
}

DATASET_SPLIT_TO_ANNOTATION_PATH = {
    DatasetSplit.TRAIN: os.path.join(DATASET_PATH, 'train_sample_data.txt'),
    DatasetSplit.VALIDATION: os.path.join(DATASET_PATH, 'validation_sample_data.txt'),
    DatasetSplit.TEST: os.path.join(DATASET_PATH, 'test_sample_data.txt'),
}
