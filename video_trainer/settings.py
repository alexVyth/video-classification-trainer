import os

DATASET_PATH = os.path.join('..', 'dataset')

FRAMES_RGB_TEMPLATE = 'frame_{0:06d}.jpg'

SAMPLE_DURATION_IN_FRAMES = 15
VIDEO_DURATION_IN_SECONDS = 300

FPS = 25

TRAIN_ANNOTATION_PATH = os.path.join(DATASET_PATH, 'train_sample_data.txt')
VALIDATION_ANNOTATION_PATH = os.path.join(DATASET_PATH, 'validation_sample_data.txt')
TEST_ANNOTATION_PATH = os.path.join(DATASET_PATH, 'test_sample_data.txt')
