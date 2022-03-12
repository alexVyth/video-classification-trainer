import math
import os
from dataclasses import dataclass
from typing import List

import cv2 as cv
import numpy
from numpy.typing import ArrayLike
from torch.utils.data import Dataset

from data.video_shift_sync import VIDEO_METADATA, VideoData
from video_trainer.loading.load_video import load_video_clip

ANNOTATION_COLOR_TO_CATEGORY = {
    0: 1,
    95: 2,
    147: 3,
    241: 4,
    101: 5,
    50: 0,
}


@dataclass
class VideoSample:
    dataset: str
    video_id: str
    start_frame: int
    label: int


class FstDataset(Dataset):
    def __init__(
        self,
        duration: int = 11,
    ):
        self.duration = duration
        self.sample_median_frame = math.ceil(self.duration / 2)
        self.samples = self._create_samples()

    def __getitem__(self, index: int) -> ArrayLike:
        sample = self.samples[index]
        video = load_video_clip(
            sample=sample, start_frame=sample.start_frame, duration=self.duration
        )
        return video, sample.label

    def __len__(self) -> int:
        return len(self.samples)

    def _create_samples(self) -> List[ArrayLike]:
        samples = []
        for video_metadata in VIDEO_METADATA:
            if video_metadata.dataset == 'OLD':
                first_index = video_metadata.first_frame
                last_index = video_metadata.last_frame - self.duration

                annotation = self._get_annotation(video_metadata)

                for frame in range(first_index, last_index, self.duration):
                    category = annotation[frame + self.sample_median_frame]
                    label = ANNOTATION_COLOR_TO_CATEGORY[category]
                    video_sample = VideoSample(
                        dataset=video_metadata.dataset,
                        video_id=video_metadata.id,
                        start_frame=frame,
                        label=label,
                    )
                    samples.append(video_sample)
        return samples

    def _get_annotation(self, video_metadata: VideoData) -> List[int]:
        if video_metadata.dataset == 'OLD':
            label_dir = os.path.join(
                '..', 'dataset', video_metadata.dataset, 'labels', f'{video_metadata.id}-NK.png'
            )
        else:
            label_dir = os.path.join(
                '..', 'dataset', video_metadata.dataset, 'labels', f'{video_metadata.id}.png'
            )

        return self._preprocess_annotation(label_dir, video_metadata)

    def _preprocess_annotation(self, label_dir: ArrayLike, video_metadata: VideoData) -> List[int]:
        annotated_frames = 300 * video_metadata.fps
        annotation = cv.imread(label_dir, 0)[25, :]
        annotation = cv.resize(
            annotation, dsize=(1, annotated_frames), interpolation=cv.INTER_NEAREST
        )[:, 0]
        annotation = self._shift_array(annotation, video_metadata.first_frame)
        return annotation  # type: ignore

    @staticmethod
    def _shift_array(array: ArrayLike, shift_magnitude: int) -> ArrayLike:
        return numpy.concatenate((numpy.zeros(shift_magnitude, dtype=numpy.uint8), array))
