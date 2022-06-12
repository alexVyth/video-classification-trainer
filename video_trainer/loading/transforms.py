from typing import Any, List

import torch
from PIL import Image
from torchvision import transforms


class ImgListToTensor(torch.nn.Module):
    @staticmethod
    def forward(
        img_list: List[Image.Image],
    ) -> torch.Tensor:
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])


class ConvertBCHWtoCBHW(torch.nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)"""

    @staticmethod
    def forward(vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)


class NoneTransform:
    def __call__(self, transform_input: Any) -> Any:
        return transform_input
