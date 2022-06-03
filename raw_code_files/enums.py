from enum import Enum


class Dataset(Enum):
    ELIDEK = 'ELIDEK'
    OLD = 'OLD'


class DatasetSplit(Enum):
    TRAIN = 'TRAIN'
    VALIDATION = 'VALIDATION'
    TEST = 'TEST'


class FstCategory(Enum):
    CLIMBING = 0
    SWIMMING = 1
    IMMOBILITY = 2
    DIVING = 3
    HEAD_SHAKE = 4
