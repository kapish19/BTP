# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import sys
import time
# sys.path.append('/home/zhaohuazhong/DGReID/')
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageOps
from reidutils.file_io import PathManager
from collections import defaultdict
import random
import torch

from .occlusion import SyntheticOcclusionTransform, apply_target_occlusion

# Legacy alias for backward compatibility
SyntheticOcclusionAugment = SyntheticOcclusionTransform

def resolve_image_path(file_name):
    if os.path.exists(file_name):
        return file_name

    fname = os.path.basename(file_name)
    cur_dir = os.path.dirname(file_name)
    parent_dir = os.path.dirname(cur_dir)

    # 1. Flat in parent dir (strip subfolder)
    p_flat = os.path.join(parent_dir, fname)
    if os.path.exists(p_flat):
        return p_flat

    # 2. Check sibling directories
    cands = ['train_v2', 'train', 'mask_train_v2', 'mask_train', 'test_v2', 'test', 'images_train']
    for cf in cands:
        p1 = os.path.join(parent_dir, cf, fname)
        if os.path.exists(p1):
            return p1
        sub = fname.split('_')[0]
        p2 = os.path.join(parent_dir, cf, sub, fname)
        if os.path.exists(p2):
            return p2

    # 3. Check grandparent directory
    grandparent_dir = os.path.dirname(parent_dir)
    for cf in cands:
        p1 = os.path.join(grandparent_dir, cf, fname)
        if os.path.exists(p1):
            return p1
        sub = fname.split('_')[0]
        p2 = os.path.join(grandparent_dir, cf, sub, fname)
        if os.path.exists(p2):
            return p2

    return file_name


def read_image(file_name, format=None):
    file_name = str(file_name).replace('\\', '/').replace('data/./data/', 'data/').replace('data/data/', 'data/').replace('./data/', 'data/')
    file_name = resolve_image_path(file_name)

    image = None
    try:
        with PathManager.open(file_name, "rb") as f:
            image = Image.open(f)
            image.load()
    except Exception:
        # Fallback to neutral placeholder if image is missing/corrupted
        image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if format is not None:
        conversion_format = format
        if format == "BGR":
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)

    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)
    elif format == "BGR":
        image = image[:, :, ::-1]
    elif len(image.shape) == 2:
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)

    image = Image.fromarray(image)
    return image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True,last_id=0, is_train=False):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        self.last_id = last_id
        self.is_train = is_train
        self.occ_aug = SyntheticOcclusionAugment(p=0.5) if is_train else None

        pid_set = set()
        pids = []
        cam_set = set()
        domains = set()
        domain = []
        dcmain = defaultdict(set)
        # self.p2d = defaultdict(set)
        for i in img_items:
            pid_set.add(i[1])
            pids.append(i[1])
            cam_set.add(i[2])
            domains.add(i[3])
            domain.append(i[3])
            dcmain[i[3]].add(i[2])
        p2d = dict(zip(pids,domain))
        domains = list(domains)
        domains.sort()

        self.demains = {}

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        self.domains = {list(domains)[i]: i for i in range(len(list(domains)))}

        self.p2d = {}

        print(self.domains)
        if relabel:
            self.pid_dict = dict([(p, i+self.last_id) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
            for p, d in p2d.items():
                self.p2d[self.pid_dict[p]] = self.domains[d]
        else:
            for p, d in p2d.items():
                self.p2d[p] = self.domains[d]
        for elm in dcmain.keys():
            self.demains[elm] = dict([(p, i) for i, p in enumerate(dcmain[elm])])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        domain = img_item[3]
        
        try:
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
        except Exception:
            # Fallback to blank tensor on any unexpected error
            img = torch.zeros(3, 256, 128)

        if self.is_train and self.occ_aug is not None:
            img, occ_mask, _ = self.occ_aug(img)
        else:
            occ_mask = torch.ones(3, 1, dtype=torch.float32)

        if self.relabel:
            pid_agg = self.pid_dict[pid]
            rescamid = self.cam_dict[camid]
            return img, pid_agg, rescamid, 1, img_path, self.domains[domain], self.demains[domain][camid], occ_mask
        else:
            return img, pid, camid, 1, img_path, self.domains[domain], self.demains[domain][camid], occ_mask

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        trackid = 1
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, img_path.split('/')[-1]


if __name__ == "__main__":
    read_image('/data/zhz_dataset/ReID/cuhk03/images_labeled/1_001_2_08.png')
