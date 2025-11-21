# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple
from torchvision.datasets import VisionDataset
from .extended import ExtendedVisionDataset
from .decoders import TargetDecoder, ImageDataDecoder
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
from PIL import Image
from openslide import OpenSlide
import random
import numpy as np
import cv2
import random


import torch
import torchvision.transforms as transforms

class SlideDataset(ExtendedVisionDataset):
    def __init__(self, root, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        
        folder_path = Path(root)

        # Image extensions to look for
        image_extensions = {'.svs'}

        # Recursively find all image files
        self.image_files_svs = [p for p in folder_path.rglob("*") if p.suffix.lower() in image_extensions]
        print("Found this many files", len(self.image_files_svs))
        
        #Load dataset_listed, which contains our acceptable patch indexes
        #/data/TCGA/575011dc-d267-4cb7-9ba2-e4d4c3c60f75/TCGA-CV-6959-01Z-00-DX1.AEC54909-0A3A-43E2-966C-7410CD7488EF.svs 13568 48128 0
        #Format is path, index, index, level
        #
        """
        self.levels = [[],[],[],[]]
        self.used_levels = [[],[],[],[]]
        with open("patches_listed", "r") as f:
            for line in f.readlines():
                parts = line.split(" ")
                path = parts[0]
                x = path[1]
                y = path[2]
                level = path[3]
                
                #We don't count this
                if level >=4:
                    pass
                self.levels[level].append((path, x, y))
        """
        self.image_files = []

        #Finishg copying from the .py
        with open("sample_dataset_30.txt", "r") as f:
            for line in f.readlines():
                self.image_files.append(line)

        print("This many resolved paths", len(self.image_files))



    #Takes a pil version of the highest level.
    def pseudo_unet(self, img, comp_w, comp_h):
       
        crop_height = comp_h
        crop_width = comp_w

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
        #210 is the invere for how strong a pixel is - 255 is pure white.
        #In our case, we have a lot of light data, which we *do* want to work with anyway
        if False:
            cv2.imwrite("binary.png", thresh)
            cv2.imwrite("gray.png", gray)
            #exit()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
        valid_components = []
        for i in range(1, num_labels):
            # The bounding box is at stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
            # stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Filter out components that are too small
            #Since we could be grabbing a piece at 224x224, ??
            if w >= crop_width and h >= crop_height:
                if False:
                    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(thresh_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imwrite(str(i) + "bounding.png", thresh_color)
                valid_components.append({'x': x, 'y': y, 'w': w, 'h': h})

        if not valid_components:
            #print("No valid components found to crop from.")
            return None

        return valid_components

    def get_all(self, index):

        path = self.image_files_svs[index]
        image = OpenSlide(path)
        return image, path

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        debug = False

        if False:#Speed test
            
            test_data = torch.randn((3, 224, 224))
            return self.transforms(transforms.ToPILImage()(test_data), None), index

        if True:
            path = self.image_files[index]
            
            #/data/TCGA/575011dc-d267-4cb7-9ba2-e4d4c3c60f75/TCGA-CV-6959-01Z-00-DX1.AEC54909-0A3A-43E2-966C-7410CD7488EF.svs 13645 48161 0
            path, x, y, level = path.split(" ")
            x = int(x)
            y = int(y)
            level = int(level)

            image = OpenSlide(path)

            patch_size = 224
            height = image.level_dimensions[0][1]
            width = image.level_dimensions[0][0]
    
            #read_region is based on the top left pixel in the level 0, not our current level
            patch = image.read_region((x, y), level=level, size=(patch_size, patch_size))

            

        #The transform used is a torchvision StandardTransform.
        #This means that it takes as input two things, and runs two different transforms on both.
        #Patch is a pillow image.

        res = patch.convert("RGB")#Removes alpha - not sure this is the best way to do this thuogh
        if self.transforms is not None:
            return self.transforms(res, None), index

        return res, None, index
        
    def hsv(self, tile_rgb, patch_size):
        
        #tile_rgb.save("tile.png")


        tile = np.array(tile_rgb)
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        min_ratio = .6
        
        lower_bound = np.array([90, 8, 103])
        upper_bound = np.array([180, 255, 255])

        mask = cv2.inRange(tile, lower_bound, upper_bound)

        ratio = np.count_nonzero(mask) / mask.size
        if ratio > min_ratio:
            #print("accept this")
            #tile_rgb.show()
            return tile_rgb
        else:
            #tile_rgb.show()
            #print("Ratio fail", ratio)
            return None

    def __len__(self) -> int:
        return len(self.image_files)

