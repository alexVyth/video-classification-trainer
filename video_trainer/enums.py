from enum import Enum


class ScoredDataset(Enum):
    FOUR_CYCLE = 'FOUR_CYCLE'


class UnscoredDataset(Enum):
    AROMATASE_PRETEST = 'AROMATASE_PRETEST'
    FOUR_CYCLE_PRETEST = 'FOUR_CYCLE_PRETEST'
    PGP_PRETEST = 'PGP_PRETEST'
    ELIDEK_PRETEST = 'ELIDEK_PRETEST'


class DatasetSplit(Enum):
    TRAIN = 'TRAIN'
    VALIDATION = 'VALIDATION'


class FstCategory(Enum):
    CLIMBING = 0
    SWIMMING = 1
    IMMOBILITY = 2
    DIVING = 3
    HEAD_SHAKE = 4
