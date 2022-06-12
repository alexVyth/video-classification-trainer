import os

from video_trainer.enums import DatasetSplit
from video_trainer.settings import TEMP_DIR

DATASET_SPLIT_TO_ANNOTATION_PATH = {
    DatasetSplit.TRAIN: os.path.join(TEMP_DIR, 'train_sample_data.txt'),
    DatasetSplit.VALIDATION: os.path.join(TEMP_DIR, 'validation_sample_data.txt'),
}
