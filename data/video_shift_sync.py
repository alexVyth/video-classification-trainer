from dataclasses import dataclass


@dataclass
class VideoData:
    dataset: str
    id: str
    first_frame: int
    fps: int

    @property
    def last_frame(self) -> int:
        return self.first_frame + (300 * self.fps)


VIDEO_METADATA = [
    VideoData('ELIDEK', '1', 625, 25),
    VideoData('ELIDEK', '2', 510, 25),
    VideoData('ELIDEK', '3', 900, 25),
    VideoData('ELIDEK', '9', 1045, 25),
    VideoData('ELIDEK', '10', 780, 25),
    VideoData('ELIDEK', '22', 750, 25),
    VideoData('ELIDEK', '24', 750, 25),
    VideoData('ELIDEK', '25', 720, 25),
]
