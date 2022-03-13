import os

from data.splitting import TEST_VIDEOS, TRAIN_VIDEOS, VALIDATION_VIDEOS
from video_trainer.enums import DatasetSplit

DATASET_PATH = os.path.join('..', 'dataset')

FRAMES_RGB_TEMPLATE = 'frame_{0:06d}.jpg'

SAMPLE_DURATION_IN_FRAMES = 15
VIDEO_DURATION_IN_SECONDS = 300

FPS = 25

DATASET_SPLIT_TO_ANNOTATION_PATH = {
    DatasetSplit.TRAIN: os.path.join(DATASET_PATH, 'train_sample_data.txt'),
    DatasetSplit.VALIDATION: os.path.join(DATASET_PATH, 'validation_sample_data.txt'),
    DatasetSplit.TEST: os.path.join(DATASET_PATH, 'test_sample_data.txt'),
}

DATASET_SPLIT_TO_VIDEOS = {
    DatasetSplit.TRAIN: TRAIN_VIDEOS,
    DatasetSplit.VALIDATION: VALIDATION_VIDEOS,
    DatasetSplit.TEST: TEST_VIDEOS,
}
