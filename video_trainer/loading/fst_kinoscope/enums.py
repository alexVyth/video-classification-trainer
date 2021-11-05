from enum import Enum


class FstCategories(Enum):
    IMMOBILITY = 1
    SWIMMING = 2
    CLIMBING = 3
    DIVING = 4
    HEAD_SHAKING = 5


FST_CATEGORIES_COLOR_MAPPING = {
    (0, 0, 255): FstCategories.IMMOBILITY,
    (255, 0, 0): FstCategories.SWIMMING,
    (0, 0, 0): FstCategories.CLIMBING,
    (255, 255, 0): FstCategories.DIVING,
    (0, 255, 0): FstCategories.HEAD_SHAKING,
}
