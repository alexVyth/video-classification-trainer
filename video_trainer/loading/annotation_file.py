import os
from dataclasses import dataclass
from math import ceil
from typing import List

import cv2 as cv
import numpy
from numpy.typing import ArrayLike

from data.video_shift_sync import VIDEO_METADATA, VideoData
from video_trainer.enums import FstCategory
from video_trainer.settings import FPS, SAMPLE_DURATION_IN_FRAMES, VIDEO_DURATION_IN_SECONDS

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


def create_samples(duration: int) -> List[VideoSample]:
    samples = []
    for video_metadata in VIDEO_METADATA:
        first_sample_start = video_metadata.first_frame
        last_sample_start = video_metadata.last_frame - duration

        label_image = _get_annotation(video_metadata)

        file_path = os.path.join(video_metadata.dataset.value, 'frames_rgb', video_metadata.name)
        for frame in range(first_sample_start, last_sample_start, duration):
            category = label_image[frame + MEDIAN_FRAME]
            label = ANNOTATION_COLOR_TO_CATEGORY[category]
            video_sample = VideoSample(file_path, frame, frame + duration, label)
            samples.append(video_sample)
    return samples


def _get_annotation(video_metadata: VideoData) -> List[int]:
    label_dir = os.path.join(
        '..', 'dataset', video_metadata.dataset.value, 'labels', f'{video_metadata.name}'
    )
    if video_metadata.dataset.value == 'OLD':
        label_dir = f'{label_dir}-NK'
    return _preprocess_annotation(f'{label_dir}.png', video_metadata)


def _preprocess_annotation(label_dir: ArrayLike, video_metadata: VideoData) -> List[int]:
    annotated_frames = VIDEO_DURATION_IN_SECONDS * FPS
    annotation = cv.imread(label_dir, 0)[25, :]
    annotation = cv.resize(
        annotation, dsize=(1, annotated_frames), interpolation=cv.INTER_NEAREST
    )[:, 0]
    annotation = _shift_array(annotation, video_metadata.first_frame)
    return annotation  # type: ignore


def _shift_array(array: ArrayLike, shift_magnitude: int) -> ArrayLike:
    return numpy.concatenate((numpy.zeros(shift_magnitude, dtype=numpy.uint8), array))


def create(annotation_file: str) -> None:
    samples = create_samples(SAMPLE_DURATION_IN_FRAMES)
    with open(annotation_file, mode='w', encoding='utf-8') as f:
        for sample in samples:
            video_id = f'{str(sample.video_path)}'
            annotation_string = (
                f'{video_id} {sample.first_frame} {sample.end_frame} {sample.label}\n'
            )
            f.write(annotation_string)