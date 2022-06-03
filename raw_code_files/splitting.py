# from random import sample
import os

from video_metadata import VIDEO_METADATA
from enums import DatasetSplit
from settings import TEMP_DIR

VIDEOS = {video.name for video in VIDEO_METADATA}
# test_videos = sample(VIDEOS, round(len(VIDEO_METADATA)* 0.2))

TEST_VIDEOS = {
     '110',
    '4C-M15',
    '4C-F5',
    '4C-F10',
    '4C-2012-F29',
    '85',
    '67',
    '4C-F16',
    '73',
    '33',
    '42',
    '4C-2012-F35',
    '62',
    '64',
    '4C-2012-F30',
    '26',
    '107',
    '4C-F2',
    '103',
    '12',
    '4C-2012-F24',
    '4C-2012-F9',
    '10',
    '4C-F48',
    '109',
    '4C-F19',
    '38',
    '4',
    '47',
    '77',
    '52',
    '102',
    '79',
    '69',
    '70',
    '4C-M36',
    '78',
    '4C-F11',
    '4C-F17',
    '4C-F24',
}

TRAIN_VALIDATION_VIDEOS = VIDEOS - TEST_VIDEOS

# validation_videos = sample(TRAIN_VALIDATION_VIDEOS, round(len(VIDEO_METADATA)* 0.2))

VALIDATION_VIDEOS = {
   '17',
    '4C-F43',
    '4C-2012-F20',
    '4C-M7',
    '40',
    '4C-2012-F21',
    '4C-F9',
    '4C-M1',
    '90',
    '68',
    '4C-2012-F5',
    '4C-F21',
    '4C-2012-F8',
    '87',
    '63',
    '27',
    '4C-F38',
    '4C-2012-F19',
    '31',
    '41',
    '4C-F40',
    '65',
    '84',
    '4C-2012-F14',
    '4C-2012-F7',
    '106',
    '9',
    '4C-2012-F1',
    '4C-2012-F27',
    '4C-2012-F22',
    '4C-M29',
    '4C-M21',
    '23',
    '58',
    '4C-F1',
    '34',
    '30',
    '4C-F28',
    '105',
    '15',
}
TRAIN_VIDEOS = TRAIN_VALIDATION_VIDEOS - VALIDATION_VIDEOS

DATASET_SPLIT_TO_VIDEOS = {
    DatasetSplit.TRAIN: TRAIN_VIDEOS,
    DatasetSplit.VALIDATION: VALIDATION_VIDEOS,
    DatasetSplit.TEST: TEST_VIDEOS,
}

DATASET_SPLIT_TO_ANNOTATION_PATH = {
    DatasetSplit.TRAIN: os.path.join(TEMP_DIR, 'train_sample_data.txt'),
    DatasetSplit.VALIDATION: os.path.join(TEMP_DIR, 'validation_sample_data.txt'),
    DatasetSplit.TEST: os.path.join(TEMP_DIR, 'test_sample_data.txt'),
}
