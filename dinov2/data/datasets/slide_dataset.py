# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Tuple
from .extended import ExtendedVisionDataset
from pathlib import Path
from openslide import OpenSlide
import numpy as np
import cv2
import random

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
        image = OpenSlide(path)
        return image, path

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = self.image_files[index]
        parts = path.split(" ")
        path, x, y, level = parts
        x = int(x)
        y = int(y)
        level = int(level)

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
        
    def hsv(self, tile_rgb, patch_size) -> Optional[Any]:
        """Filter patches using probabilistic HSV-based tissue detection.

        Instead of a hard accept/reject threshold, patches with a tissue ratio
        between ``low_ratio`` and ``high_ratio`` are accepted with probability
        equal to their ratio.  This biases the dataset toward informative
        patches while still allowing occasional lighter/borderline tiles
        through, rather than discarding them entirely.

        Acceptance rules:
          ratio >= high_ratio  → always accept  (p = 1.0)
          low_ratio <= ratio < high_ratio → accept with p = ratio
          ratio < low_ratio    → always reject  (p = 0.0)
        """
        low_ratio = 0.10
        high_ratio = 0.60

        tile = np.array(tile_rgb)
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)

        lower_bound = np.array([90, 8, 103])
        upper_bound = np.array([180, 255, 255])

        mask = cv2.inRange(tile, lower_bound, upper_bound)
        ratio = np.count_nonzero(mask) / mask.size

        if ratio >= high_ratio:
            return tile_rgb
        elif ratio >= low_ratio:
            # Probabilistic acceptance: p = ratio
            if random.random() < ratio:
                return tile_rgb
            return None
        else:
            return None

    def __len__(self) -> int:
        return len(self.image_files)
