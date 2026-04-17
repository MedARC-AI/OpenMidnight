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
from skimage import color as skimage_color
from skimage.color import rgb2hed, hed2rgb
from einops import rearrange
from PIL import Image
from turbojpeg import TurboJPEG, TJPF_RGB


from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)

logger = logging.getLogger("dinov2")


class RandStainNA(torch.nn.Module):
    """
    RandStainNA: Random Stain Normalization and Augmentation.
    Bridges stain normalization and augmentation by constraining variable stain styles
    in a practicable range using virtual template generation.

    Based on: "RandStainNA: Learning Stain-Agnostic Features from Histology Slides
    by Bridging Stain Augmentation and Normalization" (MICCAI 2022)

    Reference: https://github.com/yiqings/RandStainNA
    """

    # Default statistics for LAB color space from CRC dataset
    # These can be overridden by providing a yaml_file
    DEFAULT_LAB_STATS = {
        'L': {'avg': {'mean': 158.033, 'std': 48.792}, 'std': {'mean': 36.899, 'std': 14.383}},
        'A': {'avg': {'mean': 151.187, 'std': 10.958}, 'std': {'mean': 8.134, 'std': 2.822}},
        'B': {'avg': {'mean': 116.812, 'std': 6.643}, 'std': {'mean': 6.129, 'std': 2.013}},
    }

    # Default statistics for HED color space
    DEFAULT_HED_STATS = {
        'H': {'avg': {'mean': 0.05, 'std': 0.02}, 'std': {'mean': 0.03, 'std': 0.01}},
        'E': {'avg': {'mean': 0.02, 'std': 0.01}, 'std': {'mean': 0.02, 'std': 0.008}},
        'D': {'avg': {'mean': 0.0, 'std': 0.005}, 'std': {'mean': 0.01, 'std': 0.005}},
    }

    def __init__(
        self,
        color_space='LAB',
        std_hyper=-0.3,
        distribution='normal',
        probability=1.0,
    ):
        super().__init__()

        assert distribution in ['normal', 'laplace', 'uniform'], \
            f"Unsupported distribution: {distribution}"
        assert color_space in ['LAB', 'HSV', 'HED'], \
            f"Unsupported color space: {color_space}"

        self.color_space = color_space
        self.std_hyper = std_hyper
        self.distribution = distribution
        self.probability = probability

        if color_space == 'LAB':
            stats = self.DEFAULT_LAB_STATS
            self.channels = ['L', 'A', 'B']
        elif color_space == 'HED':
            stats = self.DEFAULT_HED_STATS
            self.channels = ['H', 'E', 'D']
        else:
            stats = self.DEFAULT_LAB_STATS
            self.channels = ['H', 'S', 'V']

        self.channel_avgs_mean = [stats[c]['avg']['mean'] for c in self.channels]
        self.channel_avgs_std = [stats[c]['avg']['std'] for c in self.channels]
        self.channel_stds_mean = [stats[c]['std']['mean'] for c in self.channels]
        self.channel_stds_std = [stats[c]['std']['std'] for c in self.channels]

    def _getavgstd(self, image):
        """Get mean and std for each channel."""
        avgs = []
        stds = []
        for idx in range(image.shape[2]):
            avgs.append(np.mean(image[:, :, idx]))
            stds.append(np.std(image[:, :, idx]))
        return np.array(avgs), np.array(stds)

    def _normalize(self, img, img_avgs, img_stds, tar_avgs, tar_stds):
        """Normalize image to target statistics."""
        img_stds = np.clip(img_stds, 0.0001, 255)
        img = (img - img_avgs) * (tar_stds / img_stds) + tar_avgs

        if self.color_space in ['LAB', 'HSV']:
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def _generate_virtual_template(self):
        """Generate virtual template statistics based on distribution."""
        tar_avgs = []
        tar_stds = []

        if self.distribution == 'uniform':
            for idx in range(3):
                tar_avg = np.random.uniform(
                    low=self.channel_avgs_mean[idx] - 3 * self.channel_avgs_std[idx],
                    high=self.channel_avgs_mean[idx] + 3 * self.channel_avgs_std[idx],
                )
                tar_std = np.random.uniform(
                    low=self.channel_stds_mean[idx] - 3 * self.channel_stds_std[idx],
                    high=self.channel_stds_mean[idx] + 3 * self.channel_stds_std[idx],
                )
                tar_avgs.append(tar_avg)
                tar_stds.append(tar_std)
        else:
            if self.distribution == 'normal':
                np_distribution = np.random.normal
            else:
                np_distribution = np.random.laplace

            for idx in range(3):
                tar_avg = np_distribution(
                    loc=self.channel_avgs_mean[idx],
                    scale=self.channel_avgs_std[idx] * (1 + self.std_hyper),
                )
                tar_std = np_distribution(
                    loc=self.channel_stds_mean[idx],
                    scale=self.channel_stds_std[idx] * (1 + self.std_hyper),
                )
                tar_avgs.append(tar_avg)
                tar_stds.append(tar_std)

        return np.array(tar_avgs), np.array(tar_stds)

    def augment(self, img):
        """Apply stain augmentation."""
        if isinstance(img, Image.Image):
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            was_pil = True
        else:
            image = img
            was_pil = False

        # Color space conversion
        if self.color_space == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif self.color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == 'HED':
            image = skimage_color.rgb2hed(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Generate virtual template and normalize
        tar_avgs, tar_stds = self._generate_virtual_template()
        img_avgs, img_stds = self._getavgstd(image)

        image = self._normalize(
            img=image,
            img_avgs=img_avgs,
            img_stds=img_stds,
            tar_avgs=tar_avgs,
            tar_stds=tar_stds,
        )

        # Convert back to BGR/RGB
        if self.color_space == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        elif self.color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.color_space == 'HED':
            nimg = skimage_color.hed2rgb(image)
            imin = nimg.min()
            imax = nimg.max()
            rsimg = (255 * (nimg - imin) / (imax - imin + 1e-8)).astype('uint8')
            image = cv2.cvtColor(rsimg, cv2.COLOR_RGB2BGR)

        # Convert back to RGB for PIL
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if was_pil:
            return Image.fromarray(image)
        return image

    def forward(self, img):
        if random.random() > self.probability:
            return img
        return self.augment(img)

class ElasticDeformation(torch.nn.Module):
    def __init__(self, low_alpha=40.0, high_alpha=200.0, low_sigma=5.0, high_sigma=10.0, probability=0.5):
        super().__init__()
        self.low_alpha = low_alpha
        self.high_alpha = high_alpha
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.probability = probability

    def forward(self, img):
        if random.random() > self.probability:
            return img


        alpha = random.uniform(self.low_alpha, self.high_alpha)
        sigma = random.uniform(self.low_sigma, self.high_sigma)


        if isinstance(img, Image.Image):
            img_np = np.array(img)
            input_type = 'PIL'
        elif isinstance(img, torch.Tensor):
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            input_type = 'Tensor'
        else:
            img_np = np.array(img)
            input_type = 'Array'
            
        h, w = img_np.shape[:2]
        
        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
                              
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        
        distorted = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        
        if input_type == 'PIL':
            return Image.fromarray(distorted)
        elif input_type == 'Tensor':
            return torch.from_numpy(distorted).permute(2, 0, 1).float() / 255.0
        return distorted


class JpegCompressionTurbo(torch.nn.Module):
    def __init__(self, quality_lower=5, quality_upper=100, probability=0.5):
        super().__init__()
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
        self.probability = probability
        
        # Initialize the C-engine once to avoid overhead during loops
        try:
            self.jpeg = TurboJPEG()
        except Exception as e:
            raise RuntimeError("TurboJPEG library not found. Run: sudo apt-get install libturbojpeg0-dev") from e

    def forward(self, img):
        if random.random() > self.probability:
            return img

        quality = random.randint(self.quality_lower, self.quality_upper)

        was_pil = False
        was_tensor = False
        
        if isinstance(img, Image.Image):
            img_np = np.array(img)
            was_pil = True
        elif isinstance(img, torch.Tensor):
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            was_tensor = True
        else:
            img_np = img

        encoded_buffer = self.jpeg.encode(img_np, quality=quality, pixel_format=TJPF_RGB)

        decoded_np = self.jpeg.decode(encoded_buffer, pixel_format=TJPF_RGB)

        if was_pil:
            return Image.fromarray(decoded_np)
        elif was_tensor:
            return torch.from_numpy(decoded_np).permute(2, 0, 1).float() / 255.0
        
        return decoded_np

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

    def forward(self, img, label=None):
        if random.random() > self.probability:
            return img

        if img is not None:
            img = torchvision.transforms.functional.pil_to_tensor(img)
            img = rearrange(img, 'c h w -> h w c')
            hed_image = rgb2hed(img)

            hed_image[..., 0] += random.uniform(self.mini, self.maxi)  # H
            hed_image[..., 1] += random.uniform(self.mini, self.maxi)  # E
            hed_image[..., 2] += random.uniform(self.mini, self.maxi)  # D

            hed_image = np.clip(hed_image, 0, 1)
            img = hed2rgb(hed_image)

            img = rearrange(img, 'h w c -> c h w')
            img = torch.from_numpy(img)
            img = torchvision.transforms.functional.to_pil_image(img)

        if label is not None:
            label = rearrange(label, 'c h w -> h w c')
            hed_image = rgb2hed(label)
            hed_image[..., 0] += random.uniform(self.mini, self.maxi)
            hed_image[..., 1] += random.uniform(self.mini, self.maxi)
            hed_image[..., 2] += random.uniform(self.mini, self.maxi)
            label = rearrange(label, 'h w c -> c h w')
            label = torch.from_numpy(label)
            return img, label

        return img


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

    Includes pathology-specific augmentations:
    - RandStainNA: Stain normalization/augmentation in LAB color space
    - HED augmentation: Color space perturbation
    - 90-degree rotations: Rotation invariance
    - Vertical and horizontal flips
    - Gaussian blur
    - Color jitter (no grayscale for H&E images)
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        ablation_mode="baseline", # This controls the abalation choices (baseline only applies the RandomResizeCrop + Horizontal Flip)
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

        # Everything is OFF by default. Only RandomResizedCrop and H-Flip are applied.
        vflip_p = 0.0
        rotate90_p = 0.0
        cj_params = None      # (brightness, contrast, saturation, hue)
        gb_params = None      # (kernel_size, sigma_min, sigma_max)
        hed_range = None      
        rs_hyper = None       
        use_elastic = False
        use_jpeg = False
        # use_ect = False

        if ablation_mode == "baseline":
            pass
        elif ablation_mode == "vflip":
            vflip_p = 0.5
        elif ablation_mode == "rotate90":
            rotate90_p = 0.5
            
        # Color Jitter Ablations
        elif ablation_mode == "colorjitter_weak":
            cj_params = (0.1, 0.1, 0.05, 0.02)
        elif ablation_mode == "colorjitter_medium":
            cj_params = (0.2, 0.2, 0.1, 0.05)
        elif ablation_mode == "colorjitter_strong":
            cj_params = (0.4, 0.4, 0.2, 0.1)
            
        # Gaussian Blur Ablations
        elif ablation_mode == "blur_weak":
            gb_params = (7, 0.3, 0.8)
        elif ablation_mode == "blur_medium":
            gb_params = (9, 0.5, 1)
        elif ablation_mode == "blur_strong":
            gb_params = (15, 1.5, 2.5)
            
        # HED Ablations
        elif ablation_mode == "hed_weak":
            hed_range = 0.005
        elif ablation_mode == "hed_medium":
            hed_range = 0.01
        elif ablation_mode == "hed_strong":
            hed_range = 0.03
            
        # RandStainNA Ablations
        elif ablation_mode == "randstain_weak":
            rs_hyper = -0.4
        elif ablation_mode == "randstain_medium":
            rs_hyper = -0.1
        elif ablation_mode == "randstain_strong":
            rs_hyper = 0.0
            
        # Combo Ablations (randstain + hed)
        elif ablation_mode == "randstain_hed_combo":
            rs_hyper = -0.3
            hed_range = 0.08
            
        elif ablation_mode == "elastic":
            use_elastic = True
            
        elif ablation_mode == "jpeg":
            use_jpeg = True
            
        elif ablation_mode == "combo":
            # Combo of all above augmentations that showed improvements in isolation
            vflip_p = 0.5
            rotate90_p = 0.5
            cj_params = (0.2, 0.2, 0.1, 0.05) 
            gb_params = (9, 0.1, 2.0)
            hed_range = 0.05
            rs_hyper = -0.3
            use_elastic = True
            use_jpeg = True
        else:
            raise ValueError(f"Unknown ablation mode: {ablation_mode}")

        # --- 3. LOGGING THE ACTIVE PIPELINE ---
        logger.info("=========================================")
        logger.info(f" BUILDING PIPELINE: MODE '{ablation_mode.upper()}' ")
        logger.info("=========================================")
        logger.info(f" [ON] RandomResizedCrop (Global: {global_crops_size}, Local: {local_crops_size})")
        logger.info(f" [ON] HorizontalFlip (p=0.5)")
        logger.info(f" [{'ON' if vflip_p > 0 else 'OFF'}] VerticalFlip (p={vflip_p})")
        logger.info(f"[{'ON' if rotate90_p > 0 else 'OFF'}] Rotate90 (p={rotate90_p})")
        
        if cj_params:
            logger.info(f" [ON] ColorJitter (B:{cj_params[0]}, C:{cj_params[1]}, S:{cj_params[2]}, H:{cj_params[3]})")
        else:
            logger.info(f" [OFF] ColorJitter")
            
        if gb_params:
            logger.info(f" [ON] GaussianBlur (Kernel:{gb_params[0]}, Sigma range: {gb_params[1]}-{gb_params[2]})")
        else:
            logger.info(f" [OFF] GaussianBlur")
            
        logger.info(f" [{'ON' if hed_range else 'OFF'}] HED Stain Perturbation (Range: +/-{hed_range})")
        logger.info(f"[{'ON' if rs_hyper is not None else 'OFF'}] RandStainNA (Std Hyper: {rs_hyper})")
        logger.info(f"[{'ON' if use_elastic else 'OFF'}] Elastic Deformation")
        logger.info(f"[{'ON' if use_jpeg else 'OFF'}] JPEG Compression")
        logger.info("=========================================\n")


        geom_global =[
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        geom_local =[
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        
        if vflip_p > 0:
            geom_global.append(transforms.RandomVerticalFlip(p=vflip_p))
            geom_local.append(transforms.RandomVerticalFlip(p=vflip_p))
        if rotate90_p > 0:
            geom_global.append(RandomRotation90()) 
            geom_local.append(RandomRotation90())
            
        self.geometric_augmentation_global = transforms.Compose(geom_global)
        self.geometric_augmentation_local = transforms.Compose(geom_local)

        def build_pixel_pipeline(blur_probability):
            pipeline =[]
            
            if use_elastic:
                pipeline.append(ElasticDeformation(probability=0.5))
                
            if rs_hyper is not None:
                pipeline.append(RandStainNA(color_space='LAB', std_hyper=rs_hyper, distribution='normal', probability=0.5))
                
            if hed_range is not None:
                pipeline.append(hed_mod(probability=0.5, perturbation_range=hed_range))
                
            if cj_params is not None:
                pipeline.append(transforms.RandomApply([transforms.ColorJitter(*cj_params)], p=0.8))
                pipeline.append(transforms.RandomGrayscale(p=0.2))
                
            if gb_params is not None:
                k, s_min, s_max = gb_params
                pipeline.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=k, sigma=(s_min, s_max))], p=blur_probability))
                            
            if use_jpeg:
                pipeline.append(JpegCompressionTurbo(quality_lower=5, quality_upper=100, probability=0.5))
                
            pipeline.extend([
                transforms.ToTensor(),
                make_normalize_transform(),
            ])
            return transforms.Compose(pipeline)

        # DINOv2 uses asymmetric blur probabilities for the different crops
        self.global_transfo1 = build_pixel_pipeline(blur_probability=1.0)
        self.global_transfo2 = build_pixel_pipeline(blur_probability=0.1)
        self.local_transfo = build_pixel_pipeline(blur_probability=0.5)
        
        # Normalization (ImageNet stats used by default)
        self.normalize = transforms.Compose([transforms.ToTensor(), make_normalize_transform()])

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