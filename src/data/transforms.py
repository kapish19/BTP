"""
Image transformations and preprocessing for Person Re-Identification.
Designed to work with standard 256x128 resolution and CLIP normalization constants.
Provides pure PyTorch / PIL implementations with zero hard external dependencies.
"""

import random
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch


# OpenAI CLIP normalization parameters
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img: Image.Image) -> torch.Tensor:
        for t in self.transforms:
            img = t(img)
        return img


class Resize:
    """Resize PIL Image to (height, width)."""
    def __init__(self, size: Tuple[int, int] = (256, 128), interpolation=Image.BICUBIC):
        self.size = size  # (H, W)
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        # PIL expects (width, height)
        return img.resize((self.size[1], self.size[0]), resample=self.interpolation)


class RandomHorizontalFlip:
    """Horizontally flip PIL Image randomly with probability p."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class PadAndRandomCrop:
    """Pad image and randomly crop back to target size."""
    def __init__(self, target_size: Tuple[int, int] = (256, 128), padding: int = 10):
        self.target_size = target_size
        self.padding = padding

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        # Create padded canvas
        padded_w = w + 2 * self.padding
        padded_h = h + 2 * self.padding
        padded_img = Image.new(img.mode, (padded_w, padded_h), (0, 0, 0))
        padded_img.paste(img, (self.padding, self.padding))

        # Random crop
        max_x = padded_w - self.target_size[1]
        max_y = padded_h - self.target_size[0]
        crop_x = random.randint(0, max(0, max_x))
        crop_y = random.randint(0, max(0, max_y))

        return padded_img.crop((crop_x, crop_y, crop_x + self.target_size[1], crop_y + self.target_size[0]))


class ToTensor:
    """Convert a PIL Image or numpy.ndarray to torch.FloatTensor in range [0.0, 1.0]."""
    def __call__(self, pic: Image.Image) -> torch.Tensor:
        if isinstance(pic, Image.Image):
            if pic.mode != 'RGB':
                pic = pic.convert('RGB')
            arr = np.array(pic, dtype=np.float32)
            # (H, W, C) -> (C, H, W)
            arr = arr.transpose((2, 0, 1)) / 255.0
            return torch.from_numpy(arr)
        elif isinstance(pic, np.ndarray):
            if pic.ndim == 2:
                pic = pic[:, :, None]
            arr = pic.transpose((2, 0, 1)).astype(np.float32) / 255.0
            return torch.from_numpy(arr)
        elif isinstance(pic, torch.Tensor):
            return pic.float()
        raise TypeError(f"Unsupported type {type(pic)}")


class Normalize:
    """Normalize a tensor image with mean and standard deviation."""
    def __init__(self, mean: List[float] = CLIP_MEAN, std: List[float] = CLIP_STD):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean.to(tensor.device)) / self.std.to(tensor.device)


def build_transforms(
    img_size: Tuple[int, int] = (256, 128),
    is_train: bool = True,
    mean: List[float] = CLIP_MEAN,
    std: List[float] = CLIP_STD
) -> Compose:
    """
    Build standard training or evaluation transform pipeline.
    """
    if is_train:
        transform_list = [
            Resize(img_size),
            RandomHorizontalFlip(p=0.5),
            PadAndRandomCrop(target_size=img_size, padding=10),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ]
    else:
        transform_list = [
            Resize(img_size),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ]
    return Compose(transform_list)
