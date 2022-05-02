import os
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import List

import cv2 as cv
import numpy
from numpy.typing import ArrayLike

from video_trainer.data.splitting import DATASET_SPLIT_TO_ANNOTATION_PATH, DATASET_SPLIT_TO_VIDEOS
from video_trainer.data.video_metadata import VIDEO_METADATA, VideoData
from video_trainer.enums import DatasetSplit, FstCategory
from video_trainer.settings import (
    FPS,
    SAMPLE_DURATION_IN_FRAMES,
    TEMP_DIR,
    VIDEO_DURATION_IN_SECONDS,
)

MEDIAN_FRAME = ceil(SAMPLE_DURATION_IN_FRAMES / 2)

ANNOTATION_COLOR_TO_CATEGORY = {
    0: FstCategory.CLIMBING.value,
    95: FstCategory.IMMOBILITY.value,
    147: FstCategory.SWIMMING.value,
    241: FstCategory.DIVING.value,
    101: FstCategory.HEAD_SHAKE.value,
}


@dataclass
class VideoSample:
    video_path: str
    first_frame: int
    end_frame: int
    label: int


def create_samples(dataset_split: DatasetSplit) -> List[VideoSample]:
    samples = []
    split_videos = DATASET_SPLIT_TO_VIDEOS[dataset_split]
    for video_metadata in VIDEO_METADATA:
        if video_metadata.name in split_videos:
            first_sample_starting_frame = video_metadata.first_annotated_frame
            last_sample_starting_frame = video_metadata.last_frame - SAMPLE_DURATION_IN_FRAMES

            label_image = _get_annotation(video_metadata)

            file_path = os.path.join(
                video_metadata.dataset.value, 'frames_rgb', video_metadata.name
            )
            for frame in range(
                first_sample_starting_frame, last_sample_starting_frame, SAMPLE_DURATION_IN_FRAMES
            ):
                category = label_image[frame + MEDIAN_FRAME]
                label = ANNOTATION_COLOR_TO_CATEGORY[category]
                video_sample = VideoSample(
                    file_path, frame, frame + SAMPLE_DURATION_IN_FRAMES - 1, label
                )
                samples.append(video_sample)
    return samples


def _get_annotation(video_metadata: VideoData) -> List[int]:
    label_dir = os.path.join(
        '..',
        'dataset',
        video_metadata.dataset.value,
        'labels',
        f'{video_metadata.name}-{video_metadata.annotator}',
    )
    return _preprocess_annotation(f'{label_dir}.png', video_metadata)


def _preprocess_annotation(label_dir: ArrayLike, video_metadata: VideoData) -> List[int]:
    annotated_frames = VIDEO_DURATION_IN_SECONDS * FPS
    annotation = cv.imread(label_dir, 0)[25, :]
    annotation = cv.resize(
        annotation, dsize=(1, annotated_frames), interpolation=cv.INTER_NEAREST
    )[:, 0]
    annotation = _shift_array(annotation, video_metadata.first_annotated_frame)
    return annotation  # type: ignore


def _shift_array(array: ArrayLike, shift_magnitude: int) -> ArrayLike:
    return numpy.concatenate((numpy.zeros(shift_magnitude, dtype=numpy.uint8), array))


def create(dataset_split: DatasetSplit) -> None:
    annotation_file = DATASET_SPLIT_TO_ANNOTATION_PATH[dataset_split]
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

    samples = create_samples(dataset_split)
    with open(annotation_file, mode='w', encoding='utf-8') as f:
        for sample in samples:
            video_id = f'{str(sample.video_path)}'
            annotation_string = (
                f'{video_id} {sample.first_frame} {sample.end_frame} {sample.label}\n'
            )
            f.write(annotation_string)
