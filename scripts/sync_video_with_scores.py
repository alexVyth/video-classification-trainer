import cv2 as cv
import numpy
from numpy.typing import ArrayLike

from video_trainer.enums import Dataset
from video_trainer.settings import FPS

DATASET = Dataset.ELIDEK
VIDEO_ID = '1'
ANNOTATION_FRAMES_SHIFT = 575

STARTING_FRAME_INDEX = 575

ANNOTATION_COLOR_TO_CATEGORY = {
    0: 'Climbing',
    95: 'Immobility',
    147: 'Swimming',
    241: 'Diving',
    50: 'Not Scored',
}

CATEGORY_TO_TEXT_COLOR = {
    'Climbing': (0, 0, 0),
    'Immobility': (255, 0, 0),
    'Swimming': (0, 0, 255),
    'Diving': (0, 255, 255),
    'Not Scored': (0, 250, 0),
}


def draw_category_to_image(image: ArrayLike, text: str) -> ArrayLike:
    return cv.putText(
        img=image,
        text=text,
        org=(0, 100),
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        color=CATEGORY_TO_TEXT_COLOR[text],
        thickness=5,
        lineType=cv.LINE_AA,
    )


def shift_array(array: ArrayLike, shift_magnitude: int) -> ArrayLike:
    return numpy.r_[numpy.full(shift_magnitude, 50), array[:-shift_magnitude]]


def preprocess_annotation(annotation_directory: ArrayLike) -> ArrayLike:
    annotated_frames = 300 * FPS

    annotation = cv.imread(annotation_directory, 0)[25, :]
    annotation = cv.resize(
        annotation, dsize=(1, annotated_frames), interpolation=cv.INTER_NEAREST
    )[:, 0]
    annotation = shift_array(annotation, ANNOTATION_FRAMES_SHIFT)
    return annotation


def main() -> None:
    video_directory = f'../dataset/{DATASET.value}/input/{VIDEO_ID}.mp4'
    annotation_directory = f'../dataset/{DATASET.value}/output/{VIDEO_ID}.png'

    annotation = preprocess_annotation(annotation_directory)

    cap = cv.VideoCapture(video_directory)
    frame_index = 0

    cv.namedWindow('Display', cv.WINDOW_NORMAL)

    while True:
        frame_index += 1
        is_successful_read, frame = cap.read()

        if not is_successful_read:
            break

        if frame_index < STARTING_FRAME_INDEX:
            continue

        category = ANNOTATION_COLOR_TO_CATEGORY[annotation[frame_index]]
        draw_category_to_image(frame, category)
        cv.imshow('Display', frame)
        if cv.waitKey(int(1000 / FPS)) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
