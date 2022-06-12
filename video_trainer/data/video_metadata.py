from dataclasses import dataclass, field
from typing import List, Tuple, Union

from video_trainer.enums import DatasetSplit, ScoredDataset, UnscoredDataset
from video_trainer.settings import FPS, VIDEO_DURATION_IN_SECONDS


@dataclass
class VideoData:
    dataset: Union[ScoredDataset, UnscoredDataset]
    name: str
    last_video_frame: int
    first_frame: List[int] = field(default_factory=lambda: [1])
    dataset_split: DatasetSplit = DatasetSplit.TRAIN

    @property
    def last_frame(self) -> int:
        return self.last_video_frame - 1


@dataclass
class ScoredData(VideoData):
    annotators: Tuple[str, str] = ('NK', 'CD')

    @property
    def first_annotated_frame(self) -> List[int]:
        return [
            round(self.first_frame[0] / 2),
            round(self.first_frame[1] / 2),
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
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F1',
        first_frame=[125, 175],
        last_video_frame=8327,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F2',
        first_frame=[225, 255],
        last_video_frame=7667,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F3',
        first_frame=[0, 31],
        last_video_frame=7560,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F4',
        first_frame=[100, 100],
        last_video_frame=7560,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F5',
        first_frame=[25, 85],
        last_video_frame=7620,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F6',
        first_frame=[25, 25],
        last_video_frame=7572,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F7',
        first_frame=[1750, 1800],
        last_video_frame=8327,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F8',
        first_frame=[50, 75],
        last_video_frame=7572,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F9',
        first_frame=[125, 175],
        last_video_frame=7596,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F10',
        first_frame=[125, 125],
        last_video_frame=7620,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F12',
        first_frame=[175, 100],
        last_video_frame=7595,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F13',
        first_frame=[200, 250],
        last_video_frame=7607,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F14',
        first_frame=[50, 65],
        last_video_frame=7597,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F15',
        first_frame=[125, 125],
        last_video_frame=7632,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F16',
        first_frame=[50, 50],
        last_video_frame=7596,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F17',
        first_frame=[100, 125],
        last_video_frame=7572,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F18',
        first_frame=[75, 50],
        last_video_frame=7596,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F19',
        first_frame=[75, 90],
        last_video_frame=7572,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F20',
        first_frame=[225, 200],
        last_video_frame=7621,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F21',
        first_frame=[225, 250],
        last_video_frame=7621,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F22',
        first_frame=[175, 150],
        last_video_frame=7632,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F23',
        first_frame=[125, 100],
        last_video_frame=7632,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F24',
        first_frame=[100, 100],
        last_video_frame=7632,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F25',
        first_frame=[50, 225],
        last_video_frame=7594,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F27',
        first_frame=[80, 50],
        last_video_frame=7561,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F29',
        first_frame=[140, 150],
        last_video_frame=7573,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F30',
        first_frame=[275, 275],
        last_video_frame=7632,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F31',
        first_frame=[200, 150],
        last_video_frame=7825,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F32',
        first_frame=[0, 35],
        last_video_frame=7608,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F34',
        first_frame=[75, 135],
        last_video_frame=7573,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F35',
        first_frame=[100, 145],
        last_video_frame=7561,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F38',
        first_frame=[100, 100],
        last_video_frame=7560,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F40',
        first_frame=[50, 75],
        last_video_frame=7535,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-2012-F41',
        first_frame=[50, 50],
        last_video_frame=7596,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F1', first_frame=[150, 25], last_video_frame=7572),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F2', first_frame=[100, 100], last_video_frame=7573),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F3', first_frame=[150, 130], last_video_frame=7548),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F4',
        first_frame=[50, 80],
        last_video_frame=7573,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F5', first_frame=[100, 55], last_video_frame=7573),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F6', first_frame=[75, 50], last_video_frame=7537),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F7',
        first_frame=[100, 75],
        last_video_frame=7669,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F8',
        first_frame=[50, 1],
        last_video_frame=7632,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F9', first_frame=[50, 50], last_video_frame=7536),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F10', first_frame=[80, 50], last_video_frame=7549),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F11', first_frame=[65, 50], last_video_frame=7561),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F12', first_frame=[1, 1], last_video_frame=7548),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F13',
        first_frame=[100, 100],
        last_video_frame=7560,
    ),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F14', first_frame=[1, 1], last_video_frame=7824),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F15', first_frame=[50, 50], last_video_frame=7537),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F16', first_frame=[75, 75], last_video_frame=7537),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F17', first_frame=[50, 100], last_video_frame=7632),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F18',
        first_frame=[175, 100],
        last_video_frame=7536,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F19', first_frame=[25, 25], last_video_frame=7573),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F20', first_frame=[50, 25], last_video_frame=7537),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F21',
        first_frame=[250, 200],
        last_video_frame=7572,
    ),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F22', first_frame=[145, 75], last_video_frame=7573),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F23',
        first_frame=[100, 500],
        last_video_frame=7704,
    ),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F24', first_frame=[100, 50], last_video_frame=7644),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F25', first_frame=[25, 1], last_video_frame=7693),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F26', first_frame=[75, 50], last_video_frame=7561),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F27', first_frame=[75, 50], last_video_frame=7632),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F28',
        first_frame=[175, 150],
        last_video_frame=7549,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F29',
        first_frame=[50, 50],
        last_video_frame=7584,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F30',
        first_frame=[25, 50],
        last_video_frame=7609,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F31',
        first_frame=[75, 75],
        last_video_frame=7621,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F32',
        first_frame=[100, 325],
        last_video_frame=7585,
    ),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-F33', first_frame=[50, 50], last_video_frame=7669),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F34',
        first_frame=[175, 150],
        last_video_frame=7692,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F35',
        first_frame=[325, 300],
        last_video_frame=7668,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F36',
        first_frame=[375, 410],
        last_video_frame=7716,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F37',
        first_frame=[75, 50],
        last_video_frame=7609,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F38',
        first_frame=[200, 240],
        last_video_frame=7608,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F39',
        first_frame=[225, 175],
        last_video_frame=7621,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F40',
        first_frame=[250, 280],
        last_video_frame=7621,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F41',
        first_frame=[325, 325],
        last_video_frame=8497,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F42',
        first_frame=[275, 305],
        last_video_frame=7645,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F43',
        first_frame=[200, 200],
        last_video_frame=7608,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F44',
        first_frame=[250, 265],
        last_video_frame=7644,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F45',
        first_frame=[150, 100],
        last_video_frame=7597,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F46',
        first_frame=[125, 215],
        last_video_frame=7705,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F47',
        first_frame=[225, 200],
        last_video_frame=7621,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-F48',
        first_frame=[375, 350],
        last_video_frame=7693,
    ),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-M1', first_frame=[300, 330], last_video_frame=7560),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-M3',
        first_frame=[225, 225],
        last_video_frame=13921,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-M5',
        first_frame=[850, 870],
        last_video_frame=8064,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-M7',
        first_frame=[150, 150],
        last_video_frame=7681,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-M11',
        first_frame=[275, 250],
        last_video_frame=7644,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-M15',
        first_frame=[300, 300],
        last_video_frame=7680,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-M17',
        first_frame=[200, 400],
        last_video_frame=7656,
    ),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-M19', first_frame=[50, 110], last_video_frame=7765),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-M21',
        first_frame=[150, 150],
        last_video_frame=7693,
        dataset_split=DatasetSplit.VALIDATION,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-M23',
        first_frame=[100, 180],
        last_video_frame=7705,
    ),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-M25', first_frame=[50, 180], last_video_frame=7632),
    ScoredData(ScoredDataset.FOUR_CYCLE, '4C-M27', first_frame=[100, 50], last_video_frame=7645),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-M29',
        first_frame=[175, 215],
        last_video_frame=7609,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-M31',
        first_frame=[225, 260],
        last_video_frame=7609,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-M33',
        first_frame=[100, 125],
        last_video_frame=7561,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-M35',
        first_frame=[200, 200],
        last_video_frame=7608,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-M36',
        first_frame=[275, 250],
        last_video_frame=7741,
    ),
    ScoredData(
        ScoredDataset.FOUR_CYCLE,
        '4C-M37',
        first_frame=[150, 150],
        last_video_frame=7572,
    ),
]
