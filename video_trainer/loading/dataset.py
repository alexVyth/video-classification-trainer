import math
import os
from dataclasses import dataclass

import cv2 as cv
import numpy
from numpy.typing import ArrayLike
from torch.utils.data import Dataset

from data.video_shift_sync import VIDEO_SYNC
from video_trainer.loading.load_video import load_video_clip

ANNOTATION_COLOR_TO_CATEGORY = {
    0: '1',
    95: '2',
    147: '3',
    241: '4',
    50: '0',
}


@dataclass
class VideoSample:
    video_id: int
    start_frame: int
    label: int


def shift_array(array: ArrayLike, shift_magnitude: int) -> ArrayLike:
    return numpy.concatenate((numpy.zeros(shift_magnitude, dtype=numpy.uint8), array))


class FstDataset(Dataset):
    def __init__(
        self,
        directory: str = os.path.join('..', 'dataset', 'ELIDEK'),
        duration: int = 11,
    ):
        self.video_directory = os.path.join(directory, 'videos')
        self.labels_directory = os.path.join(directory, 'labels')
        self.duration = duration
        self.fps = 25
        self.samples = self._create_samples()

    def __getitem__(self, index: int) -> ArrayLike:
        sample = self.samples[index]
        video = load_video_clip(
            video_id=sample.video_id, start_frame=sample.start_frame, duration=self.duration
        )
        return video, sample.label

    def __len__(self) -> int:
        return len(self.samples)

    def _preprocess_annotation(self, label_directory: ArrayLike, shift: int) -> ArrayLike:
        annotated_frames = 300 * self.fps
        annotation = cv.imread(label_directory, 0)[25, :]
        annotation = cv.resize(
            annotation, dsize=(1, annotated_frames), interpolation=cv.INTER_NEAREST
        )[:, 0]
        annotation = shift_array(annotation, shift)
        return annotation

    def _create_samples(self) -> ArrayLike:
        samples = []
        for video_id in VIDEO_SYNC.keys():
            label_path = f'../dataset/ELIDEK/labels/{video_id}.png'
            first_frame = VIDEO_SYNC[video_id]
            last_frame = first_frame + 300 * self.fps - self.duration
            annotation = self._preprocess_annotation(label_path, first_frame)
            label_index = math.ceil(self.duration)
            for frame in range(first_frame, last_frame, self.duration):
                category = annotation[frame + label_index]
                label = ANNOTATION_COLOR_TO_CATEGORY[category]
                video_object = VideoSample(
                    video_id=int(video_id),
                    start_frame=frame,
                    label=int(label),
                )
                samples.append(video_object)
        return samples
