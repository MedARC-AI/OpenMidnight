# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple
from .extended import ExtendedVisionDataset
from pathlib import Path
from openslide import OpenSlide
import numpy as np
import cv2
from PIL import Image

class SlideDataset(ExtendedVisionDataset):
    def __init__(self, root, sample_list_path, *args, patch_size_pixels=224, **kwargs) -> None:
        super().__init__(root, *args, **kwargs)
        self.sample_list_path = Path(sample_list_path)
        self.patch_size_pixels = patch_size_pixels
        if not self.sample_list_path.is_file():
            raise FileNotFoundError(f"Sample list not found at {self.sample_list_path}")

        with self.sample_list_path.open("r") as f:
            self.image_files = [line.strip() for line in f if line.strip()]

        print(f"This many resolved paths {len(self.image_files)} from {self.sample_list_path} (patch_size={patch_size_pixels})")

    def get_all(self, index):
        parts = self.image_files[index].split(" ")
        path = parts[0]
        image = OpenSlide(path)
        return image, path

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        parts = self.image_files[index].split(" ")
        path = parts[0]
        x = int(parts[1])
        y = int(parts[2])
        level = int(parts[3])
        read_size = int(parts[4]) if len(parts) >= 5 else self.patch_size_pixels

        slide = OpenSlide(path)
        patch = slide.read_region((x, y), level=level, size=(read_size, read_size))
        res = patch.convert("RGB")
        if read_size != self.patch_size_pixels:
            res = res.resize((self.patch_size_pixels, self.patch_size_pixels), Image.BICUBIC)

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
