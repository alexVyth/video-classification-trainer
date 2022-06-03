from dataclasses import dataclass

from enums import Dataset
from settings import FPS, VIDEO_DURATION_IN_SECONDS


@dataclass
class VideoData:
    dataset: Dataset
    name: str
    first_annotated_frame: int
    last_video_frame: int
    annotator: str

    @property
    def last_frame(self) -> int:
        return min(
            [
                self.first_annotated_frame + (VIDEO_DURATION_IN_SECONDS * FPS),
                self.last_video_frame - 1,
            ]
        )


VIDEO_METADATA = [
    VideoData(Dataset.ELIDEK, '1', 625, 8496, 'TM'),
    VideoData(Dataset.ELIDEK, '2', 510, 8496, 'TM'),
    VideoData(Dataset.ELIDEK, '3', 900, 9030, 'TM'),
    VideoData(Dataset.ELIDEK, '4', 800, 9031, 'TM'),
    VideoData(Dataset.ELIDEK, '5', 1150, 9277, 'TM'),
    VideoData(Dataset.ELIDEK, '6', 1150, 9277, 'TM'),
    VideoData(Dataset.ELIDEK, '7', 500, 8957, 'TM'),
    VideoData(Dataset.ELIDEK, '8', 550, 8957, 'TM'),
    VideoData(Dataset.ELIDEK, '9', 1045, 9769, 'TM'),
    VideoData(Dataset.ELIDEK, '10', 780, 9769, 'TM'),
    VideoData(Dataset.ELIDEK, '11', 450, 7993, 'TM'),
    VideoData(Dataset.ELIDEK, '12', 200, 7993, 'TM'),
    VideoData(Dataset.ELIDEK, '22', 750, 8718, 'TM'),
    VideoData(Dataset.ELIDEK, '24', 750, 8518, 'TM'),
    VideoData(Dataset.ELIDEK, '25', 720, 9122, 'TM'),
    VideoData(Dataset.ELIDEK, '14', 775, 8012, 'TM'),
    VideoData(Dataset.ELIDEK, '15', 525, 8208, 'TM'),
    VideoData(Dataset.ELIDEK, '16', 525, 8208, 'TM'),
    VideoData(Dataset.ELIDEK, '17', 550, 8093, 'TM'),
    VideoData(Dataset.ELIDEK, '18', 550, 8093, 'TM'),
    VideoData(Dataset.ELIDEK, '19', 650, 8210, 'TM'),
    VideoData(Dataset.ELIDEK, '20', 650, 8210, 'TM'),
    VideoData(Dataset.ELIDEK, '21', 625, 8719, 'TM'),
    VideoData(Dataset.ELIDEK, '23', 575, 8519, 'TM'),
    VideoData(Dataset.ELIDEK, '26', 675, 9123, 'TM'),
    VideoData(Dataset.ELIDEK, '27', 725, 8526, 'TM'),
    VideoData(Dataset.ELIDEK, '28', 725, 8526, 'TM'),
    VideoData(Dataset.ELIDEK, '29', 775, 8954, 'TM'),
    VideoData(Dataset.ELIDEK, '30', 775, 8954, 'TM'),
    VideoData(Dataset.ELIDEK, '31', 650, 8672, 'TM'),
    VideoData(Dataset.ELIDEK, '32', 650, 8672, 'TM'),
    VideoData(Dataset.ELIDEK, '33', 450, 8103, 'TM'),
    VideoData(Dataset.ELIDEK, '34', 450, 8103, 'TM'),
    VideoData(Dataset.ELIDEK, '35', 550, 8299, 'TM'),
    VideoData(Dataset.ELIDEK, '36', 550, 8299, 'TM'),
    VideoData(Dataset.ELIDEK, '37', 475, 8198, 'TM'),
    VideoData(Dataset.ELIDEK, '38', 475, 8198, 'TM'),
    VideoData(Dataset.ELIDEK, '39', 650, 8439, 'TM'),
    VideoData(Dataset.ELIDEK, '40', 650, 8439, 'TM'),
    VideoData(Dataset.ELIDEK, '41', 650, 9479, 'TM'),
    VideoData(Dataset.ELIDEK, '42', 650, 9479, 'TM'),
    VideoData(Dataset.ELIDEK, '43', 600, 8607, 'TM'),
    VideoData(Dataset.ELIDEK, '44', 600, 8607, 'TM'),
    VideoData(Dataset.ELIDEK, '45', 750, 8667, 'TM'),
    VideoData(Dataset.ELIDEK, '46', 750, 8667, 'TM'),
    VideoData(Dataset.ELIDEK, '47', 1425, 9781, 'TM'),
    VideoData(Dataset.ELIDEK, '48', 1425, 9781, 'TM'),
    VideoData(Dataset.ELIDEK, '49', 750, 8695, 'TM'),
    VideoData(Dataset.ELIDEK, '50', 750, 8695, 'TM'),
    VideoData(Dataset.ELIDEK, '51', 550, 8322, 'TM'),
    VideoData(Dataset.ELIDEK, '52', 550, 8322, 'TM'),
    VideoData(Dataset.ELIDEK, '53', 650, 8555, 'TM'),
    VideoData(Dataset.ELIDEK, '54', 650, 8555, 'TM'),
    VideoData(Dataset.ELIDEK, '55', 550, 8327, 'TM'),
    VideoData(Dataset.ELIDEK, '56', 550, 8327, 'TM'),
    VideoData(Dataset.ELIDEK, '57', 2725, 10840, 'TM'),
    VideoData(Dataset.ELIDEK, '58', 2725, 10840, 'TM'),
    VideoData(Dataset.ELIDEK, '59', 550, 8389, 'TM'),
    VideoData(Dataset.ELIDEK, '60', 550, 8389, 'TM'),
    VideoData(Dataset.ELIDEK, '61', 650, 8693, 'TM'),
    VideoData(Dataset.ELIDEK, '62', 650, 8693, 'TM'),
    VideoData(Dataset.ELIDEK, '63', 525, 8329, 'TM'),
    VideoData(Dataset.ELIDEK, '64', 525, 8329, 'TM'),
    VideoData(Dataset.ELIDEK, '65', 550, 8399, 'TM'),
    VideoData(Dataset.ELIDEK, '66', 550, 8399, 'TM'),
    VideoData(Dataset.ELIDEK, '67', 525, 8239, 'TM'),
    VideoData(Dataset.ELIDEK, '68', 525, 8239, 'TM'),
    VideoData(Dataset.ELIDEK, '69', 575, 8386, 'TM'),
    VideoData(Dataset.ELIDEK, '70', 575, 8386, 'TM'),
    VideoData(Dataset.ELIDEK, '71', 575, 8293, 'TM'),
    VideoData(Dataset.ELIDEK, '72', 575, 8293, 'TM'),
    VideoData(Dataset.ELIDEK, '73', 500, 8253, 'TM'),
    VideoData(Dataset.ELIDEK, '74', 500, 8253, 'TM'),
    VideoData(Dataset.ELIDEK, '75', 500, 8282, 'TM'),
    VideoData(Dataset.ELIDEK, '76', 500, 8282, 'TM'),
    VideoData(Dataset.ELIDEK, '77', 675, 8486, 'TM'),
    VideoData(Dataset.ELIDEK, '78', 675, 8486, 'TM'),
    VideoData(Dataset.ELIDEK, '79', 650, 8547, 'TM'),
    VideoData(Dataset.ELIDEK, '80', 650, 8547, 'TM'),
    VideoData(Dataset.ELIDEK, '81', 275, 7788, 'TM'),
    VideoData(Dataset.ELIDEK, '82', 275, 7788, 'TM'),
    VideoData(Dataset.ELIDEK, '83', 325, 7866, 'TM'),
    VideoData(Dataset.ELIDEK, '84', 325, 7866, 'TM'),
    VideoData(Dataset.ELIDEK, '85', 350, 8094, 'TM'),
    VideoData(Dataset.ELIDEK, '86', 350, 8094, 'TM'),
    VideoData(Dataset.ELIDEK, '87', 575, 8111, 'TM'),
    VideoData(Dataset.ELIDEK, '88', 700, 8111, 'TM'),
    VideoData(Dataset.ELIDEK, '89', 500, 8253, 'TM'),
    VideoData(Dataset.ELIDEK, '90', 500, 8253, 'TM'),
    VideoData(Dataset.ELIDEK, '101', 225, 7883, 'TM'),
    VideoData(Dataset.ELIDEK, '102', 400, 7883, 'TM'),
    VideoData(Dataset.ELIDEK, '103', 200, 7953, 'TM'),
    VideoData(Dataset.ELIDEK, '104', 200, 7953, 'TM'),
    VideoData(Dataset.ELIDEK, '105', 250, 7830, 'TM'),
    VideoData(Dataset.ELIDEK, '106', 250, 7830, 'TM'),
    VideoData(Dataset.ELIDEK, '107', 500, 8321, 'TM'),
    VideoData(Dataset.ELIDEK, '108', 500, 8321, 'TM'),
    VideoData(Dataset.ELIDEK, '109', 300, 8069, 'TM'),
    VideoData(Dataset.ELIDEK, '110', 300, 8069, 'TM'),
    VideoData(Dataset.OLD, '4C-2012-F1', 125, 8327, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F2', 225, 7667, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F3', 0, 7560, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F4', 100, 7560, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F5', 25, 7620, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F6', 25, 7572, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F7', 1750, 8327, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F8', 50, 7572, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F9', 125, 7596, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F10', 125, 7620, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F12', 175, 7595, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F13', 200, 7607, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F14', 50, 7597, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F15', 125, 7632, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F16', 50, 7596, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F17', 100, 7572, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F18', 75, 7596, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F19', 75, 7572, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F20', 225, 7621, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F21', 225, 7621, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F22', 175, 7632, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F23', 125, 7632, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F24', 100, 7632, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F25', 50, 7594, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F27', 80, 7561, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F29', 140, 7573, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F30', 275, 7632, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F31', 200, 7825, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F32', 0, 7608, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F34', 75, 7573, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F35', 100, 7561, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F38', 100, 7560, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F40', 50, 7535, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F41', 50, 7596, 'NK'),
    VideoData(Dataset.OLD, '4C-F1', 150, 7572, 'NK'),
    VideoData(Dataset.OLD, '4C-F2', 100, 7573, 'NK'),
    VideoData(Dataset.OLD, '4C-F3', 150, 7548, 'NK'),
    VideoData(Dataset.OLD, '4C-F4', 50, 7573, 'NK'),
    VideoData(Dataset.OLD, '4C-F5', 100, 7573, 'NK'),
    VideoData(Dataset.OLD, '4C-F6', 75, 7537, 'NK'),
    VideoData(Dataset.OLD, '4C-F7', 100, 7669, 'NK'),
    VideoData(Dataset.OLD, '4C-F8', 50, 7632, 'NK'),
    VideoData(Dataset.OLD, '4C-F9', 50, 7536, 'NK'),
    VideoData(Dataset.OLD, '4C-F10', 80, 7549, 'NK'),
    VideoData(Dataset.OLD, '4C-F11', 65, 7561, 'NK'),
    VideoData(Dataset.OLD, '4C-F12', 0, 7548, 'NK'),
    VideoData(Dataset.OLD, '4C-F13', 101, 7560, 'NK'),
    VideoData(Dataset.OLD, '4C-F14', 0, 7824, 'NK'),
    VideoData(Dataset.OLD, '4C-F15', 50, 7537, 'NK'),
    VideoData(Dataset.OLD, '4C-F16', 75, 7537, 'NK'),
    VideoData(Dataset.OLD, '4C-F17', 50, 7632, 'NK'),
    VideoData(Dataset.OLD, '4C-F18', 175, 7536, 'NK'),
    VideoData(Dataset.OLD, '4C-F19', 25, 7573, 'NK'),
    VideoData(Dataset.OLD, '4C-F20', 50, 7537, 'NK'),
    VideoData(Dataset.OLD, '4C-F21', 250, 7572, 'NK'),
    VideoData(Dataset.OLD, '4C-F22', 145, 7573, 'NK'),
    VideoData(Dataset.OLD, '4C-F23', 100, 7704, 'NK'),
    VideoData(Dataset.OLD, '4C-F24', 100, 7644, 'NK'),
    VideoData(Dataset.OLD, '4C-F25', 25, 7693, 'NK'),
    VideoData(Dataset.OLD, '4C-F26', 75, 7561, 'NK'),
    VideoData(Dataset.OLD, '4C-F27', 75, 7632, 'NK'),
    VideoData(Dataset.OLD, '4C-F28', 175, 7549, 'NK'),
    VideoData(Dataset.OLD, '4C-F29', 50, 7584, 'NK'),
    VideoData(Dataset.OLD, '4C-F30', 25, 7609, 'NK'),
    VideoData(Dataset.OLD, '4C-F31', 75, 7621, 'NK'),
    VideoData(Dataset.OLD, '4C-F32', 100, 7585, 'NK'),
    VideoData(Dataset.OLD, '4C-F33', 50, 7669, 'NK'),
    VideoData(Dataset.OLD, '4C-F34', 175, 7692, 'NK'),
    VideoData(Dataset.OLD, '4C-F35', 325, 7668, 'NK'),
    VideoData(Dataset.OLD, '4C-F36', 375, 7716, 'NK'),
    VideoData(Dataset.OLD, '4C-F37', 75, 7609, 'NK'),
    VideoData(Dataset.OLD, '4C-F38', 200, 7608, 'NK'),
    VideoData(Dataset.OLD, '4C-F39', 225, 7621, 'NK'),
    VideoData(Dataset.OLD, '4C-F40', 250, 7621, 'NK'),
    VideoData(Dataset.OLD, '4C-F41', 325, 8497, 'NK'),
    VideoData(Dataset.OLD, '4C-F42', 275, 7645, 'NK'),
    VideoData(Dataset.OLD, '4C-F43', 200, 7608, 'NK'),
    VideoData(Dataset.OLD, '4C-F44', 250, 7644, 'NK'),
    VideoData(Dataset.OLD, '4C-F45', 150, 7597, 'NK'),
    VideoData(Dataset.OLD, '4C-F46', 125, 7705, 'NK'),
    VideoData(Dataset.OLD, '4C-F47', 225, 7621, 'NK'),
    VideoData(Dataset.OLD, '4C-F48', 375, 7693, 'NK'),
    VideoData(Dataset.OLD, '4C-M1', 300, 7560, 'NK'),
    VideoData(Dataset.OLD, '4C-M3', 225, 13921, 'NK'),
    VideoData(Dataset.OLD, '4C-M5', 850, 8064, 'NK'),
    VideoData(Dataset.OLD, '4C-M7', 150, 7681, 'NK'),
    VideoData(Dataset.OLD, '4C-M11', 275, 7644, 'NK'),
    VideoData(Dataset.OLD, '4C-M15', 300, 7680, 'NK'),
    VideoData(Dataset.OLD, '4C-M17', 200, 7656, 'NK'),
    VideoData(Dataset.OLD, '4C-M19', 50, 7765, 'NK'),
    VideoData(Dataset.OLD, '4C-M21', 150, 7693, 'NK'),
    VideoData(Dataset.OLD, '4C-M23', 100, 7705, 'NK'),
    VideoData(Dataset.OLD, '4C-M25', 50, 7632, 'NK'),
    VideoData(Dataset.OLD, '4C-M27', 100, 7645, 'NK'),
    VideoData(Dataset.OLD, '4C-M29', 175, 7609, 'NK'),
    VideoData(Dataset.OLD, '4C-M31', 225, 7609, 'NK'),
    VideoData(Dataset.OLD, '4C-M33', 100, 7561, 'NK'),
    VideoData(Dataset.OLD, '4C-M35', 200, 7608, 'NK'),
    VideoData(Dataset.OLD, '4C-M36', 275, 7741, 'NK'),
    VideoData(Dataset.OLD, '4C-M37', 150, 7572, 'NK'),
    VideoData(Dataset.OLD, '4C-2012-F1', 175, 8327, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F2', 255, 7667, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F3', 31, 7560, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F4', 100, 7560, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F5', 85, 7620, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F6', 26, 7572, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F7', 1800, 8327, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F8', 75, 7572, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F9', 175, 7596, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F10', 125, 7620, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F12', 100, 7595, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F13', 250, 7607, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F14', 65, 7597, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F15', 125, 7632, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F16', 50, 7596, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F17', 125, 7572, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F18', 50, 7596, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F19', 90, 7572, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F20', 200, 7621, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F21', 250, 7621, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F22', 150, 7632, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F23', 100, 7632, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F24', 100, 7632, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F25', 225, 7594, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F27', 50, 7561, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F29', 150, 7573, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F30', 275, 7632, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F31', 150, 7825, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F32', 35, 7608, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F34', 135, 7573, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F35', 145, 7561, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F38', 100, 7560, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F40', 75, 7535, 'CD'),
    VideoData(Dataset.OLD, '4C-2012-F41', 50, 7596, 'CD'),
    VideoData(Dataset.OLD, '4C-F1', 25, 7572, 'CD'),
    VideoData(Dataset.OLD, '4C-F2', 100, 7573, 'CD'),
    VideoData(Dataset.OLD, '4C-F3', 130, 7548, 'CD'),
    VideoData(Dataset.OLD, '4C-F4', 80, 7573, 'CD'),
    VideoData(Dataset.OLD, '4C-F5', 55, 7573, 'CD'),
    VideoData(Dataset.OLD, '4C-F6', 50, 7537, 'CD'),
    VideoData(Dataset.OLD, '4C-F7', 75, 7669, 'CD'),
    VideoData(Dataset.OLD, '4C-F8', 1, 7632, 'CD'),
    VideoData(Dataset.OLD, '4C-F9', 51, 7536, 'CD'),
    VideoData(Dataset.OLD, '4C-F10', 50, 7549, 'CD'),
    VideoData(Dataset.OLD, '4C-F11', 50, 7561, 'CD'),
    VideoData(Dataset.OLD, '4C-F12', 1, 7548, 'CD'),
    VideoData(Dataset.OLD, '4C-F13', 101, 7560, 'CD'),
    VideoData(Dataset.OLD, '4C-F14', 1, 7824, 'CD'),
    VideoData(Dataset.OLD, '4C-F15', 50, 7537, 'CD'),
    VideoData(Dataset.OLD, '4C-F16', 75, 7537, 'CD'),
    VideoData(Dataset.OLD, '4C-F17', 100, 7632, 'CD'),
    VideoData(Dataset.OLD, '4C-F18', 100, 7536, 'CD'),
    VideoData(Dataset.OLD, '4C-F19', 25, 7573, 'CD'),
    VideoData(Dataset.OLD, '4C-F20', 25, 7537, 'CD'),
    VideoData(Dataset.OLD, '4C-F21', 200, 7572, 'CD'),
    VideoData(Dataset.OLD, '4C-F22', 75, 7573, 'CD'),
    VideoData(Dataset.OLD, '4C-F23', 50, 7704, 'CD'),
    VideoData(Dataset.OLD, '4C-F24', 50, 7644, 'CD'),
    VideoData(Dataset.OLD, '4C-F25', 1, 7693, 'CD'),
    VideoData(Dataset.OLD, '4C-F26', 50, 7561, 'CD'),
    VideoData(Dataset.OLD, '4C-F27', 50, 7632, 'CD'),
    VideoData(Dataset.OLD, '4C-F28', 150, 7549, 'CD'),
    VideoData(Dataset.OLD, '4C-F29', 50, 7584, 'CD'),
    VideoData(Dataset.OLD, '4C-F30', 50, 7609, 'CD'),
    VideoData(Dataset.OLD, '4C-F31', 75, 7621, 'CD'),
    VideoData(Dataset.OLD, '4C-F32', 325, 7585, 'CD'),
    VideoData(Dataset.OLD, '4C-F33', 50, 7669, 'CD'),
    VideoData(Dataset.OLD, '4C-F34', 150, 7692, 'CD'),
    VideoData(Dataset.OLD, '4C-F35', 300, 7668, 'CD'),
    VideoData(Dataset.OLD, '4C-F36', 410, 7716, 'CD'),
    VideoData(Dataset.OLD, '4C-F37', 50, 7609, 'CD'),
    VideoData(Dataset.OLD, '4C-F38', 240, 7608, 'CD'),
    VideoData(Dataset.OLD, '4C-F39', 175, 7621, 'CD'),
    VideoData(Dataset.OLD, '4C-F40', 280, 7621, 'CD'),
    VideoData(Dataset.OLD, '4C-F41', 325, 8497, 'CD'),
    VideoData(Dataset.OLD, '4C-F42', 305, 7645, 'CD'),
    VideoData(Dataset.OLD, '4C-F43', 200, 7608, 'CD'),
    VideoData(Dataset.OLD, '4C-F44', 265, 7644, 'CD'),
    VideoData(Dataset.OLD, '4C-F45', 100, 7597, 'CD'),
    VideoData(Dataset.OLD, '4C-F46', 215, 7705, 'CD'),
    VideoData(Dataset.OLD, '4C-F47', 200, 7621, 'CD'),
    VideoData(Dataset.OLD, '4C-F48', 350, 7693, 'CD'),
    VideoData(Dataset.OLD, '4C-M1', 330, 7560, 'CD'),
    VideoData(Dataset.OLD, '4C-M3', 225, 13921, 'CD'),
    VideoData(Dataset.OLD, '4C-M5', 870, 8064, 'CD'),
    VideoData(Dataset.OLD, '4C-M7', 150, 7681, 'CD'),
    VideoData(Dataset.OLD, '4C-M11', 250, 7644, 'CD'),
    VideoData(Dataset.OLD, '4C-M15', 300, 7680, 'CD'),
    VideoData(Dataset.OLD, '4C-M17', 400, 7656, 'CD'),
    VideoData(Dataset.OLD, '4C-M19', 110, 7765, 'CD'),
    VideoData(Dataset.OLD, '4C-M21', 150, 7693, 'CD'),
    VideoData(Dataset.OLD, '4C-M23', 180, 7705, 'CD'),
    VideoData(Dataset.OLD, '4C-M25', 50, 7632, 'CD'),
    VideoData(Dataset.OLD, '4C-M27', 50, 7645, 'CD'),
    VideoData(Dataset.OLD, '4C-M29', 215, 7609, 'CD'),
    VideoData(Dataset.OLD, '4C-M31', 260, 7609, 'CD'),
    VideoData(Dataset.OLD, '4C-M33', 125, 7561, 'CD'),
    VideoData(Dataset.OLD, '4C-M35', 200, 7608, 'CD'),
    VideoData(Dataset.OLD, '4C-M36', 250, 7741, 'CD'),
    VideoData(Dataset.OLD, '4C-M37', 150, 7572, 'CD'),
]
