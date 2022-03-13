from enum import Enum


class Dataset(Enum):
    ELIDEK = 'ELIDEK'
    OLD = 'OLD'


class FstCategory(Enum):
    CLIMBING = 1
    SWIMMING = 2
    IMMOBILITY = 3
    DIVING = 4
    HEAD_SHAKE = 5
