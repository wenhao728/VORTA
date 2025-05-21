from typing import Literal

SupportedModels = Literal["hunyuan", "wan"]

STA_TILE_SIZE = [6, 8, 8]
STA_VIDEO_SIZE = [117, 768, 1280]

NAME_TO_VIDEO_SIZE = {
    'wan-1_3B': [77, 480, 832],
    'wan-14B': [77, 720, 1280],
    'hunyuan': [117, 720, 1280],
}