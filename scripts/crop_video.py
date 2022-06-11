import glob
import re
import subprocess
from typing import Tuple

import cv2
from numpy.typing import ArrayLike

from video_trainer.enums import UnscoredDataset

DATASET = UnscoredDataset.ELIDEK_PRETEST
MOUSE_POSITION = 'RIGHT'
VIDEO_FORMAT = 'MP4'


def get_video_frame(directory: str) -> ArrayLike:
    video_capture = cv2.VideoCapture(directory)
    for _ in range(1000):
        video_capture.read()
    _, image = video_capture.read()
    video_capture.release()
    return image


def get_roi(image: ArrayLike) -> Tuple[int, int, int, int]:
    roi = cv2.selectROI(image)
    cv2.destroyAllWindows()
    return roi  # type: ignore


def get_video_destination(video_source: str) -> str:
    video_destination = video_source.replace(DATASET.value, f'{DATASET.value}_CROPPED')
    pattern = r'-\d+' if MOUSE_POSITION == 'LEFT' else r'\d+-'
    return re.sub(pattern, '', video_destination)


def crop_video(video_source: str, video_destination: str, roi: Tuple[int, int, int, int]) -> None:
    x, y, width, height = roi
    command = (
        f'ffmpeg -hwaccel cuda -i {video_source} '
        f'-filter:v "fps=25, crop={width}:{height}:{x}:{y}" {video_destination}'
    )
    subprocess.call(command, shell=True)


def main() -> None:
    rois = []
    video_directories = glob.glob(
        f'../videos/{DATASET.value}/**/**.{VIDEO_FORMAT}', recursive=True
    )
    for video_source in video_directories:
        frame = get_video_frame(video_source)
        rois.append(get_roi(frame))

    for video_source, roi in zip(video_directories, rois):
        video_destination = get_video_destination(video_source)
        crop_video(video_source, video_destination, roi)


if __name__ == '__main__':
    main()
