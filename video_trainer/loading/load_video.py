import os

import torch
import torchvision


def load_video_clip(sample, start_frame: int, duration: int) -> torch.Tensor:  # type: ignore
    frames_per_second = 25
    start_time = _convert_frame_to_second(start_frame, frames_per_second)
    end_time = (
        _convert_frame_to_second(start_frame + duration, frames_per_second) - 1 / frames_per_second
    )
    video_extension = 'mp4' if sample.dataset == 'ELIDEK' else 'mkv'
    video, _, _ = torchvision.io.read_video(
        os.path.join(
            '..', 'dataset', f'{sample.dataset}', 'videos', f'{sample.video_id}.{video_extension}'
        ),
        start_time,
        end_time,
        pts_unit='sec',
    )

    return video


def _convert_frame_to_second(frame: int, frames_per_second: int) -> float:
    return frame / frames_per_second
