import os
import threading
from queue import Queue
from typing import Tuple

import cv2

from video_trainer.enums import Dataset
from video_trainer.settings import DATASET_PATH, FRAMES_RGB_TEMPLATE

DATASET = Dataset.OLD
OUT_HEIGHT_WIDTH = (112, 112)

VIDEO_EXTENSION = 'mp4' if DATASET == Dataset.ELIDEK else 'mkv'
VIDEO_PATH = os.path.join(DATASET_PATH, DATASET.value, 'videos')
FRAMES_RGB_PATH = os.path.join(DATASET_PATH, DATASET.value, 'frames_rgb')
NUM_THREADS = 4
SKIP_FRAMES = 1 if DATASET == Dataset.ELIDEK else 2


def video_to_rgb(video_filename: str, out_dir: str, resize_shape: Tuple[int, int]) -> None:
    reader = cv2.VideoCapture(video_filename)
    (
        success,
        frame,
    ) = reader.read()  # read first frame

    count = 0
    while success:
        out_filepath = os.path.join(out_dir, FRAMES_RGB_TEMPLATE.format(count))
        frame = cv2.resize(frame, resize_shape)
        cv2.imwrite(out_filepath, frame)
        for _ in range(SKIP_FRAMES):
            success, frame = reader.read()
        count += 1


def process_videofile(
    video_filename: str, video_path: str, rgb_out_path: str, file_extension: str
) -> None:
    filepath = os.path.join(video_path, video_filename)
    video_filename = video_filename.replace(file_extension, '')

    out_dir = os.path.join(rgb_out_path, video_filename)
    os.mkdir(out_dir)
    video_to_rgb(filepath, out_dir, resize_shape=OUT_HEIGHT_WIDTH)


def thread_job(
    queue: Queue, video_path: str, rgb_out_path: str, file_extension: str  # type: ignore
) -> None:
    while not queue.empty():
        video_filename = queue.get()
        process_videofile(video_filename, video_path, rgb_out_path, file_extension=file_extension)
        queue.task_done()


def main() -> None:
    video_filenames = os.listdir(VIDEO_PATH)
    queue = Queue()  # type: ignore
    _ = [queue.put(video_filename) for video_filename in video_filenames]  # type: ignore

    for _ in range(NUM_THREADS):
        worker = threading.Thread(
            target=thread_job, args=(queue, VIDEO_PATH, FRAMES_RGB_PATH, f'.{VIDEO_EXTENSION}')
        )
        worker.start()
    queue.join()


if __name__ == '__main__':
    main()
