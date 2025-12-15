# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple
from .extended import ExtendedVisionDataset
from pathlib import Path
from openslide import OpenSlide
import numpy as np
import cv2

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
        if len(parts) != 6:
            raise ValueError(f"Expected 6 fields per line, got {len(parts)}")
        path = parts[0]
        image = OpenSlide(path)
        return image, path

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = self.image_files[index]
        parts = path.split(" ")
        if len(parts) != 6:
            raise ValueError(f"Expected 6 fields per line, got {len(parts)}")
        path, x, y, level, mpp_x, mpp_y = parts
        x = int(x)
        y = int(y)
        level = int(level)
        mpp_x = float(mpp_x)
        mpp_y = float(mpp_y)

        image = OpenSlide(path)

        patch_size = 224
        height = image.level_dimensions[0][1]
        width = image.level_dimensions[0][0]

        #read_region is based on the top left pixel in the level 0, not our current level
        patch = image.read_region((x, y), level=level, size=(patch_size, patch_size))

        res = patch.convert("RGB") # Removes alpha - not sure this is the best way to do this thuogh
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
