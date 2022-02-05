import os
from dataclasses import dataclass

import torch
import torchvision

from video_trainer.loading.fst_kinoscope.enums import FstCategory


@dataclass
class VideoSample:
    video_id: int
    start_frame: int
    end_frame: int
    label: FstCategory


def convert_frame_to_second(frame: int, frames_per_second: int) -> float:
    return frame / frames_per_second


def load_video_clip(video_id: int, start_frame: int, duration: int) -> torch.Tensor:
    frames_per_second = 25
    start_time = convert_frame_to_second(start_frame, frames_per_second)
    end_time = (
        convert_frame_to_second(start_frame + duration, frames_per_second) - 1 / frames_per_second
    )
    video, _, _ = torchvision.io.read_video(
        os.path.join('/dataset', 'ELIDEK', 'inputs', f'{video_id}.MP4'),
        start_time,
        end_time,
        pts_unit='sec',
    )

    return video


def main() -> None:
    video_sample = VideoSample(video_id=2, start_frame=20, end_frame=31, label=FstCategory.DIVING)
    print(video_sample)
    print(video_sample.start_frame)

    video_id = 1
    start_frame = 10
    duration = 11
    load_video_clip(video_id, start_frame, start_frame + duration)


if __name__ == '__main__':
    main()
