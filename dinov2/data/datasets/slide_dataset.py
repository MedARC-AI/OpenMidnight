# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import atexit
from collections import OrderedDict
from typing import Any, Tuple

from .extended import ExtendedVisionDataset
from pathlib import Path
from openslide import OpenSlide
import numpy as np
import cv2

_SLIDE_CACHE: "OrderedDict[str, OpenSlide]" = OrderedDict()
_SLIDE_CACHE_LIMIT = 4096


def _get_slide(path: str) -> OpenSlide:
    """Return a cached OpenSlide handle, opening and LRU-evicting as needed."""
    slide = _SLIDE_CACHE.get(path)
    if slide is not None:
        _SLIDE_CACHE.move_to_end(path)
        return slide
    slide = OpenSlide(path)
    _SLIDE_CACHE[path] = slide
    if len(_SLIDE_CACHE) > _SLIDE_CACHE_LIMIT:
        _, old = _SLIDE_CACHE.popitem(last=False)
        old.close()
    return slide


def _close_all_slides():
    for slide in _SLIDE_CACHE.values():
        slide.close()
    _SLIDE_CACHE.clear()


atexit.register(_close_all_slides)


class SlideDataset(ExtendedVisionDataset):
    def __init__(self, root, sample_list_path, *args, **kwargs) -> None:
        super().__init__(root, *args, **kwargs)
        self.sample_list_path = Path(sample_list_path)
        if not self.sample_list_path.is_file():
            raise FileNotFoundError(f"Sample list not found at {self.sample_list_path}")

        with self.sample_list_path.open("r") as f:
            self.image_files = [line.strip() for line in f if line.strip()]

        print(f"This many resolved paths {len(self.image_files)} from {self.sample_list_path}")

    def get_all(self, index):
        parts = self.image_files[index].split(" ")
        path = parts[0]
        image = _get_slide(path)
        return image, path

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = self.image_files[index]
        parts = path.split(" ")
        path, x, y, level = parts
        x = int(x)
        y = int(y)
        level = int(level)

        image = _get_slide(path)

        patch_size = 224

        patch = image.read_region((x, y), level=level, size=(patch_size, patch_size))

        res = patch.convert("RGB")
        if self.transforms is not None:
            return self.transforms(res, None), index

        return res, None, index
        
    def hsv(self, tile_rgb, patch_size):
        tile = np.array(tile_rgb)
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        min_ratio = .6
        
        lower_bound = np.array([90, 8, 103])
        upper_bound = np.array([180, 255, 255])

        mask = cv2.inRange(tile, lower_bound, upper_bound)

        ratio = np.count_nonzero(mask) / mask.size
        if ratio > min_ratio:
            return tile_rgb
        else: # ratio failed, reject
            return None

    def __len__(self) -> int:
        return len(self.image_files)
