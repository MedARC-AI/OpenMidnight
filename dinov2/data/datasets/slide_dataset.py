# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple
from .extended import ExtendedVisionDataset
from pathlib import Path
import logging
import math
import random

from openslide import OpenSlide
from PIL import Image
import numpy as np
import cv2


logger = logging.getLogger("dinov2")

MPP_X_KEY = "openslide.mpp-x"
MPP_Y_KEY = "openslide.mpp-y"
DEFAULT_MPP = 0.25

def _normalize_tcga_project_id(project_id):
    if project_id is None:
        return None
    value = str(project_id).strip()
    if not value or value.lower() == "null":
        return None
    value = value.upper()
    if value.startswith("TCGA-"):
        return value
    return f"TCGA-{value}"


def _line_matches_tcga_project(line, candidates):
    path = line.split(" ", 1)[0]
    path_upper = path.upper()
    return any(candidate in path_upper for candidate in candidates if candidate)


def _parse_optional_float(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() == "null":
            return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("Expected a positive finite float")
    return parsed


def _parse_sample_line(line):
    parts = line.split()
    if len(parts) not in (4, 6):
        raise ValueError(f"Expected 4 or 6 fields per line, got {len(parts)}")
    path, x, y, level = parts[:4]
    mpp_x = mpp_y = None
    if len(parts) == 6:
        mpp_x = _parse_optional_float(parts[4])
        mpp_y = _parse_optional_float(parts[5])
    return path, int(x), int(y), int(level), mpp_x, mpp_y


class SlideDataset(ExtendedVisionDataset):
    def __init__(
        self,
        root,
        sample_list_path,
        tcga_project_id=None,
        mpp_min=None,
        mpp_max=None,
        global_crops_size=None,
        global_crops_scale_min=None,
        global_crops_scale_max=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(root, *args, **kwargs)
        self.sample_list_path = Path(sample_list_path)
        if not self.sample_list_path.is_file():
            raise FileNotFoundError(f"Sample list not found at {self.sample_list_path}")

        with self.sample_list_path.open("r") as f:
            self.image_files = [line.strip() for line in f if line.strip()]

        project_id = _normalize_tcga_project_id(tcga_project_id)
        if project_id:
            candidates = {project_id, project_id.replace("TCGA-", "")}
            before_count = len(self.image_files)
            filtered = [
                line for line in self.image_files if _line_matches_tcga_project(line, candidates)
            ]
            if not filtered:
                raise ValueError(
                    f"No entries matched tcga_project_id={tcga_project_id!r} in {self.sample_list_path}. "
                    "Ensure sample list paths include the project id (e.g., TCGA-BRCA)."
                )
            self.image_files = filtered
            print(
                "Filtered sample list to {} entries for tcga_project_id={} (from {})".format(
                    len(self.image_files), tcga_project_id, before_count
                )
            )

        print(f"This many resolved paths {len(self.image_files)} from {self.sample_list_path}")
        self.mpp_min = _parse_optional_float(mpp_min)
        self.mpp_max = _parse_optional_float(mpp_max)
        if self.mpp_min is not None and self.mpp_max is not None and self.mpp_min > self.mpp_max:
            raise ValueError("mpp_min must be <= mpp_max")
        self.global_crops_size = int(global_crops_size)
        self.global_crops_scale_min = float(global_crops_scale_min)
        self.global_crops_scale_max = float(global_crops_scale_max)
        if self.global_crops_size <= 0:
            raise ValueError("global_crops_size must be > 0")
        if self.global_crops_scale_min <= 0 or self.global_crops_scale_max <= 0:
            raise ValueError("global_crops_scale_min/max must be > 0")
        if self.global_crops_scale_min > self.global_crops_scale_max:
            raise ValueError("global_crops_scale_min must be <= global_crops_scale_max")
        self._global_scale_min_sqrt = math.sqrt(self.global_crops_scale_min)
        self._global_scale_max_sqrt = math.sqrt(self.global_crops_scale_max)
        if self.mpp_min is not None and self.mpp_max is not None:
            min_mpp_in = self.mpp_min / self._global_scale_max_sqrt
            max_mpp_in = self.mpp_max / self._global_scale_min_sqrt
            if min_mpp_in > max_mpp_in:
                raise ValueError("MPP_min/MPP_max incompatible with global_crops_scale")
        self._patch_size = self.global_crops_size
        self._max_skip_attempts = 20
        self._logged_mpp_skips = set()

    def get_all(self, index):
        path, *_ = _parse_sample_line(self.image_files[index])
        image = OpenSlide(path)
        return image, path

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        for attempt in range(self._max_skip_attempts):
            sample_index = index if attempt == 0 else random.randrange(len(self.image_files))
            line = self.image_files[sample_index]
            path, x, y, level, mpp_x, mpp_y = _parse_sample_line(line)

            image = OpenSlide(path)
            mpp_x, mpp_y = self._resolve_mpp_for_level(image, level, mpp_x, mpp_y)
            mpp = (mpp_x + mpp_y) / 2.0

            read_size = self._patch_size
            if self.mpp_min is not None:
                min_mpp_in = self.mpp_min / self._global_scale_max_sqrt
                if mpp < min_mpp_in:
                    required = (min_mpp_in * self._patch_size) / mpp
                    read_size = max(self._patch_size, int(math.ceil(required)))

            mpp_in = mpp * (read_size / self._patch_size)
            scale_min = self.global_crops_scale_min
            scale_max = self.global_crops_scale_max
            if self.mpp_min is not None:
                scale_min = max(scale_min, (self.mpp_min / mpp_in) ** 2)
            if self.mpp_max is not None:
                scale_max = min(scale_max, (self.mpp_max / mpp_in) ** 2)
            if scale_min > scale_max:
                reason = "final global crop mpp outside bounds; upsampling required"
                self._log_mpp_skip(path, level, reason)
                image.close()
                continue

            if not self._region_fits(image, x, y, level, read_size):
                image.close()
                continue

            # read_region is based on the top left pixel in level 0 coordinates
            patch = image.read_region((x, y), level=level, size=(read_size, read_size))
            image.close()

            res = patch.convert("RGB")  # Removes alpha
            if read_size != self._patch_size:
                res = res.resize((self._patch_size, self._patch_size), resample=Image.BICUBIC)
            if self.transforms is not None:
                meta = {"global_scale_range": (scale_min, scale_max)}
                return self.transforms((res, meta), None), sample_index

            return res, None, sample_index

        raise RuntimeError("Failed to sample a valid slide patch after multiple attempts.")
        
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

    def _resolve_mpp_for_level(self, image, level, mpp_x, mpp_y):
        if mpp_x is not None and mpp_y is not None:
            return mpp_x, mpp_y
        props = image.properties
        base_mpp_x = _parse_optional_float(props.get(MPP_X_KEY)) or DEFAULT_MPP
        base_mpp_y = _parse_optional_float(props.get(MPP_Y_KEY)) or DEFAULT_MPP
        downsample = float(image.level_downsamples[level])
        level_mpp_x = base_mpp_x * downsample
        level_mpp_y = base_mpp_y * downsample
        if mpp_x is None:
            mpp_x = level_mpp_x
        if mpp_y is None:
            mpp_y = level_mpp_y
        return mpp_x, mpp_y

    def _region_fits(self, image, x, y, level, read_size):
        downsample = float(image.level_downsamples[level])
        level0_w, level0_h = image.level_dimensions[0]
        region_w = read_size * downsample
        region_h = read_size * downsample
        return x >= 0 and y >= 0 and (x + region_w) <= level0_w and (y + region_h) <= level0_h

    def _log_mpp_skip(self, path, level, reason):
        key = (path, level, reason)
        if key in self._logged_mpp_skips:
            return
        logger.warning("Skipping sample %s level %s: %s", path, level, reason)
        self._logged_mpp_skips.add(key)
