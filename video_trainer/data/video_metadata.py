from dataclasses import dataclass
from typing import List, Tuple

from video_trainer.enums import ScoredDataset
from video_trainer.settings import FPS, VIDEO_DURATION_IN_SECONDS


@dataclass
class VideoData:
    dataset: ScoredDataset
    name: str
    _first_annotated_frame: List[int]
    last_video_frame: int
    annotators: Tuple[str, str] = ('NK', 'CD')

    @property
    def first_annotated_frame(self) -> List[int]:
        return [
            round(self._first_annotated_frame[0] / 2),
            round(self._first_annotated_frame[1] / 2),
        ]

    @property
    def last_frame(self) -> int:
        return min(
            [
                self.first_annotated_frame[0] + (VIDEO_DURATION_IN_SECONDS * FPS),
                self.first_annotated_frame[1] + (VIDEO_DURATION_IN_SECONDS * FPS),
                self.last_video_frame - 1,
            ]
        )


VIDEO_METADATA = [
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F1', [125, 175], 8327),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F2', [225, 255], 7667),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F3', [0, 31], 7560),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F4', [100, 100], 7560),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F5', [25, 85], 7620),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F6', [25, 25], 7572),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F7', [1750, 1800], 8327),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F8', [50, 75], 7572),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F9', [125, 175], 7596),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F10', [125, 125], 7620),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F12', [175, 100], 7595),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F13', [200, 250], 7607),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F14', [50, 65], 7597),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F15', [125, 125], 7632),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F16', [50, 50], 7596),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F17', [100, 125], 7572),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F18', [75, 50], 7596),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F19', [75, 90], 7572),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F20', [225, 200], 7621),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F21', [225, 250], 7621),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F22', [175, 150], 7632),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F23', [125, 100], 7632),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F24', [100, 100], 7632),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F25', [50, 225], 7594),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F27', [80, 50], 7561),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F29', [140, 150], 7573),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F30', [275, 275], 7632),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F31', [200, 150], 7825),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F32', [0, 35], 7608),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F34', [75, 135], 7573),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F35', [100, 145], 7561),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F38', [100, 100], 7560),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F40', [50, 75], 7535),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-2012-F41', [50, 50], 7596),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F1', [150, 25], 7572),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F2', [100, 100], 7573),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F3', [150, 130], 7548),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F4', [50, 80], 7573),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F5', [100, 55], 7573),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F6', [75, 50], 7537),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F7', [100, 75], 7669),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F8', [50, 1], 7632),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F9', [50, 50], 7536),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F10', [80, 50], 7549),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F11', [65, 50], 7561),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F12', [1, 1], 7548),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F13', [100, 100], 7560),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F14', [1, 1], 7824),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F15', [50, 50], 7537),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F16', [75, 75], 7537),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F17', [50, 100], 7632),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F18', [175, 100], 7536),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F19', [25, 25], 7573),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F20', [50, 25], 7537),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F21', [250, 200], 7572),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F22', [145, 75], 7573),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F23', [100, 500], 7704),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F24', [100, 50], 7644),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F25', [25, 1], 7693),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F26', [75, 50], 7561),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F27', [75, 50], 7632),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F28', [175, 150], 7549),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F29', [50, 50], 7584),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F30', [25, 50], 7609),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F31', [75, 75], 7621),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F32', [100, 325], 7585),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F33', [50, 50], 7669),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F34', [175, 150], 7692),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F35', [325, 300], 7668),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F36', [375, 410], 7716),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F37', [75, 50], 7609),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F38', [200, 240], 7608),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F39', [225, 175], 7621),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F40', [250, 280], 7621),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F41', [325, 325], 8497),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F42', [275, 305], 7645),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F43', [200, 200], 7608),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F44', [250, 265], 7644),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F45', [150, 100], 7597),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F46', [125, 215], 7705),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F47', [225, 200], 7621),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-F48', [375, 350], 7693),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M1', [300, 330], 7560),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M3', [225, 225], 13921),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M5', [850, 870], 8064),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M7', [150, 150], 7681),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M11', [275, 250], 7644),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M15', [300, 300], 7680),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M17', [200, 400], 7656),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M19', [50, 110], 7765),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M21', [150, 150], 7693),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M23', [100, 180], 7705),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M25', [50, 180], 7632),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M27', [100, 50], 7645),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M29', [175, 215], 7609),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M31', [225, 260], 7609),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M33', [100, 125], 7561),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M35', [200, 200], 7608),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M36', [275, 250], 7741),
    VideoData(ScoredDataset.FOUR_CYCLE, '4C-M37', [150, 150], 7572),
]
