import glob
import re
import subprocess
from typing import Tuple

import cv2
from numpy.typing import ArrayLike

from video_trainer.enums import Dataset

DATASET = Dataset.ELIDEK
MOUSE_POSITION = 'RIGHT'
VIDEO_FORMAT = 'MP4'


def get_video_frame(directory: str) -> ArrayLike:
    video_capture = cv2.VideoCapture(directory)
    for _ in range(1000):
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
        f'-filter:v "crop={width}:{height}:{x}:{y}" {video_destination}'
    )
    subprocess.call(command, shell=True)


def main() -> None:
    video_directories = glob.glob(
        f'../dataset/{DATASET.value}/**/**.{VIDEO_FORMAT}', recursive=True
    )
    for video_source in video_directories:
        video_destination = get_video_destination(video_source)
        frame = get_video_frame(video_source)
        roi = get_roi(frame)
        crop_video(video_source, video_destination, roi)


if __name__ == '__main__':
    main()
