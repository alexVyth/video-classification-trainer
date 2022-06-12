import os
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import List, Optional, Union

import cv2 as cv
import numpy
from numpy.typing import ArrayLike

from video_trainer.data.splitting import DATASET_SPLIT_TO_ANNOTATION_PATH
from video_trainer.data.video_metadata import ScoredData, VideoData
from video_trainer.enums import DatasetSplit, FstCategory
from video_trainer.settings import (
    DATASET_PATH,
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
    label: Optional[int]


def create(
    dataset_split: DatasetSplit,
    videos_metadata: Union[List[VideoData], List[ScoredData]],
) -> None:
    annotation_file = DATASET_SPLIT_TO_ANNOTATION_PATH[dataset_split]
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

    samples = _create_samples(dataset_split, videos_metadata)
    with open(annotation_file, mode='w', encoding='utf-8') as f:
        for sample in samples:
            video_id = f'{str(sample.video_path)}'
            annotation_string = (
                f'{video_id} {sample.first_frame} {sample.end_frame} {sample.label}\n'
            )
            f.write(annotation_string)


def _create_samples(
    dataset_split: DatasetSplit,
    videos_metadata: Union[List[VideoData], List[ScoredData]],
) -> List[VideoSample]:
    samples = []
    for video_metadata in videos_metadata:
        if video_metadata.dataset_split != dataset_split:
            continue

        first_sample_starting_frame = (
            video_metadata.first_annotated_frame if isinstance(video_metadata, ScoredData) else [1]
        )
        last_sample_starting_frame = video_metadata.last_frame - SAMPLE_DURATION_IN_FRAMES

        file_path = os.path.join(video_metadata.dataset.value, 'frames_rgb', video_metadata.name)

        if isinstance(video_metadata, ScoredData):
            label_images = _get_annotations(video_metadata)

        for frame in range(
            first_sample_starting_frame[0],
            last_sample_starting_frame,
            SAMPLE_DURATION_IN_FRAMES,
        ):
            label = None

            if isinstance(video_metadata, ScoredData):
                category_1 = label_images[0][frame + MEDIAN_FRAME]
                category_2 = label_images[1][frame + MEDIAN_FRAME]
                if category_1 != category_2:
                    continue
                label = ANNOTATION_COLOR_TO_CATEGORY[category_1]

            video_sample = VideoSample(
                file_path, frame, frame + SAMPLE_DURATION_IN_FRAMES - 1, label
            )
            samples.append(video_sample)
    return samples


def _get_annotations(video_metadata: ScoredData) -> List[List[int]]:
    label_dir_1 = os.path.join(
        DATASET_PATH,
        video_metadata.dataset.value,
        'labels',
        f'{video_metadata.name}-{video_metadata.annotators[0]}',
    )
    label_dir_2 = os.path.join(
        DATASET_PATH,
        video_metadata.dataset.value,
        'labels',
        f'{video_metadata.name}-{video_metadata.annotators[1]}',
    )
    return [
        _preprocess_annotation(f'{label_dir_1}.png', video_metadata.first_annotated_frame[0]),
        _preprocess_annotation(f'{label_dir_2}.png', video_metadata.first_annotated_frame[1]),
    ]


def _preprocess_annotation(label_dir: ArrayLike, first_annotated_frame: int) -> List[int]:
    annotated_frames = VIDEO_DURATION_IN_SECONDS * FPS
    annotation = cv.imread(label_dir, 0)[25, :]
    annotation = cv.resize(
        annotation, dsize=(1, annotated_frames), interpolation=cv.INTER_NEAREST
    )[:, 0]
    annotation = _shift_array(annotation, first_annotated_frame)
    return annotation  # type: ignore


def _shift_array(array: ArrayLike, shift_magnitude: int) -> ArrayLike:
    return numpy.concatenate((numpy.zeros(shift_magnitude, dtype=numpy.uint8), array))
