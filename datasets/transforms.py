"""
Data transforms and preprocessing pipelines for DG-ReID.
Default resolution: 256 x 128.
Standard Re-ID augmentations:
  - Random Horizontal Flip
  - Padding (10px) + Random Cropping
  - Normalization using standard CLIP visual statistics
"""

import torchvision.transforms as T
from PIL import Image

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def build_transforms(img_size=(256, 128), is_train=True, pad=10):
    """
    Builds torchvision transforms for training or evaluation.
    Args:
        img_size: tuple (height, width), default (256, 128)
        is_train: bool, True for training, False for evaluation
        pad: padding size for random crop
    """
    h, w = img_size

    if is_train:
        transform_list = [
            T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(pad, padding_mode='edge'),
            T.RandomCrop((h, w)),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ]
    else:
        transform_list = [
            T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ]

    return T.Compose(transform_list)
