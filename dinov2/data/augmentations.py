# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)
from skimage.color import rgb2hed, hed2rgb

import torch
from einops import rearrange, reduce, repeat
import random
import matplotlib.pyplot as plt
logger = logging.getLogger("dinov2")
import numpy as np
import torchvision

class hed_mod(torch.nn.Module):

    def forward(self, img, label = None):
        

        chance = random.uniform(0,1) > .5
        if chance:
            return img

        if img !=None:
            #Convert image from RGB to HED.
            #Input shape is (3,size, size)
            #Convert to chanels last, then swap back
            img = torchvision.transforms.functional.pil_to_tensor(img)

            img = rearrange(img, 'c h w -> h w c')
            img_orig = img
            hed_image = rgb2hed(img)
            #Modify channels, each with random amount, between -.05 and .05
            mini =  -.05
            maxi = .05
            if False:
                hed_image[..., 0] *= (1 + random.uniform(mini, maxi))#H
                hed_image[..., 1] *= (1 + random.uniform(mini, maxi))#E
                hed_image[..., 2] *= (1 + random.uniform(mini, maxi))#D
            else:
                hed_image[..., 0] += random.uniform(mini, maxi)#H
                hed_image[..., 1] += random.uniform(mini, maxi)#E
                hed_image[..., 2] += random.uniform(mini, maxi)#D

            #Make sure legit image
            hed_image = np.clip(hed_image, 0, 1)   
            img = hed2rgb(hed_image)

            if False:#debug
                fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # Adjust figsize as needed
                axes[0].imshow(img_orig)
                axes[0].set_title("Before")
                axes[0].axis('off') # Turn off axis ticks and labels for cleaner image display

                # Plot the "After" image on the second subplot
                axes[1].imshow(img)
                axes[1].set_title("After")
                axes[1].axis('off') # Turn off axis ticks and labels

                # Adjust layout to prevent titles from overlapping
                plt.tight_layout()

                # Set the overall figure title (optional)
                fig.suptitle("Image Comparison: Before and After HED Channel Modification", y=1.02) # y adjusts title position

                plt.show()
                
                exit()
            img = rearrange(img, 'h w c -> c h w')
            img = torch.from_numpy(img)
            img = torchvision.transforms.functional.to_pil_image(img)
        
        if label != None:
            label = rearrange(label, 'c h w -> h w c')
            hed_image = rgb2hed(label)
            #Modify channels
            hed_image[..., 0] += random.uniform(0, total) - maxi#H
            hed_image[..., 1] += random.uniform(0, total) - maxi#E
            hed_image[..., 2] += random.uniform(0, total) - maxi#D
            label = rearrange(label, 'h w c -> c h w')
            label = torch.from_numpy(label)

            return img, label 

        return img



class DataAugmentationDINO(object):
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

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    #[transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                #transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

    
        self.global_transfo1 = transforms.Compose([hed_mod(), color_jittering, global_transfo1_extra, self.normalize])#Do we apply to everything?
        self.global_transfo2 = transforms.Compose([hed_mod(), color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([hed_mod(), color_jittering, local_transfo_extra, self.normalize])

        
        #self.global_nohed = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        #self.global_nohed2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        #self.local_nohed = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

        self.hedonly = transforms.Compose([hed_mod(), self.normalize])



    def __call__(self, image):

        output = {}
        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)


        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]
        
        #print("gloabl crop shapes", global_crop_1.shape)
        #print(global_crop_2.shape)
        from torchvision.utils import save_image
        if False:#Saving
            image.save("original.png")
            save_image(self.hedonly(im1_base), "global_hed.png")
            save_image(self.hedonly(im2_base), "global2_hed.png")

            save_image(global_crop_1, "global.png")
            save_image(global_crop_2, "global2.png")
            save_image(self.global_nohed(im1_base), "global1_nohed.png")
            save_image(self.global_nohed2(im2_base), "global2_nohed.png")
            save_image(global_crop_1, "global.png")
            save_image(global_crop_2, "global2.png")
            exit()
        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        if False:
            for i, local in enumerate(local_crops):
                save_image(local, str(i) + "local" + ".png")
            local_crops_new = [
                    self.local_nohed(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
                ]

            local_crops_new2 = [
                    self.hedonly(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
                ]


            for i, local in enumerate(local_crops_new):
                save_image(local, str(i) + "localnohed.png")
            
            for i, local in enumerate(local_crops_new2):
                save_image(local, str(i) + "localhed2.png")

                
            exit()
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
