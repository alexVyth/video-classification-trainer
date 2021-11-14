from enum import Enum


class FstCategory(Enum):
    IMMOBILITY = 0
    SWIMMING = 1
    CLIMBING = 2
    DIVING = 3
    HEAD_SHAKING = 4


class Color(Enum):
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)
    YELLOW = (255, 255, 0)
    GREEN = (0, 255, 0)


FST_CATEGORIES_COLOR_MAPPING: dict[Color, FstCategory] = {
    Color.BLUE: FstCategory.IMMOBILITY,
    Color.RED: FstCategory.SWIMMING,
    Color.BLACK: FstCategory.CLIMBING,
    Color.YELLOW: FstCategory.DIVING,
    Color.GREEN: FstCategory.HEAD_SHAKING,
}


def map_color_to_category(color: Color) -> FstCategory:
    return FST_CATEGORIES_COLOR_MAPPING[color]
