# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import cv2
import numpy as np
import random
import torch
import torchvision
from torchvision import transforms
from skimage.color import rgb2hed, hed2rgb
from PIL import Image

from .transforms import (
    make_normalize_transform,
)

logger = logging.getLogger("dinov2")


class hed_mod(torch.nn.Module):
    """
    HED color space augmentation for H&E stained histopathology images.
    Randomly perturbs Hematoxylin, Eosin, and DAB channels.
    """

    def __init__(self, probability=0.5, perturbation_range=0.05):
        super().__init__()
        self.probability = probability
        self.mini = -perturbation_range
        self.maxi = perturbation_range

    def _to_float_numpy(self, pil_img):
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.expand_dims(arr, -1)
        return arr

    def _to_pil(self, np_img):
        return Image.fromarray((np_img * 255.0).clip(0, 255).astype(np.uint8))

    def forward(self, img, label=None):
        if random.random() > self.probability:
            if label is None:
                return img
            return img, label

        out_img = img
        if img is not None:
            hed_image = rgb2hed(self._to_float_numpy(img))

            hed_image[..., 0] += random.uniform(self.mini, self.maxi)  # H
            hed_image[..., 1] += random.uniform(self.mini, self.maxi)  # E
            hed_image[..., 2] += random.uniform(self.mini, self.maxi)  # D

            hed_image = np.clip(hed_image, 0, 1)
            rgb_image = np.clip(hed2rgb(hed_image), 0, 1)
            out_img = self._to_pil(rgb_image)

        if label is not None:
            hed_label = rgb2hed(self._to_float_numpy(label))
            hed_label[..., 0] += random.uniform(self.mini, self.maxi)
            hed_label[..., 1] += random.uniform(self.mini, self.maxi)
            hed_label[..., 2] += random.uniform(self.mini, self.maxi)
            hed_label = np.clip(hed_label, 0, 1)
            label_rgb = np.clip(hed2rgb(hed_label), 0, 1)
            label = self._to_pil(label_rgb)
            return out_img, label

        return out_img


class RandomRotation90(torch.nn.Module):
    """
    Random 90-degree rotation augmentation for histopathology images.
    Pathology images are rotation-invariant, so we can apply 0, 90, 180, or 270 degree rotations.
    """

    def __init__(self):
        super().__init__()
        self.angles = [0, 90, 180, 270]

    def forward(self, img):
        angle = random.choice(self.angles)
        if angle == 0:
            return img
        return transforms.functional.rotate(img, angle)


class DataAugmentationDINO(object):
    """
    Data augmentation pipeline for DINOv2 training on histopathology images.
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # Geometric augmentations with rotation and both flips
        self.geometric_augmentation_global = transforms.Compose(
            [
                # RandomRotation90(),
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                # RandomRotation90(),
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        # Normalization (ImageNet stats used by default)
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        hed_aug = hed_mod(probability=0.5, perturbation_range=0.05)

        self.global_transfo1 = transforms.Compose([
            # hed_aug,
            self.normalize
        ])

        self.global_transfo2 = transforms.Compose([
            # hed_aug,
            self.normalize
        ])

        self.local_transfo = transforms.Compose([
            # hed_aug,
            self.normalize
        ])

    def __call__(self, image):
        output = {}

        # Global crops
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # Local crops
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
