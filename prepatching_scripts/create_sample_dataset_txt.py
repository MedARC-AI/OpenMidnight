import cv2
import random
from pathlib import Path
from openslide import OpenSlide
import numpy as np

data_root = Path("/data/TCGA")
output_filename = "sample_dataset_30.txt"
patch_size = 224
max_tries = 1000

def hsv(tile_rgb):
    """
    Checks if a given tile has a high concentration of tissue based on an HSV mask.
    """
    tile = np.array(tile_rgb)
    # Convert from RGB to HSV color space
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    min_ratio = .6

    # Define the color range for tissue in HSV
    lower_bound = np.array([90, 8, 103])
    upper_bound = np.array([180, 255, 255])

    # Create a mask for the specified color range
    mask = cv2.inRange(tile, lower_bound, upper_bound)

    # Calculate the ratio of tissue pixels
    ratio = np.count_nonzero(mask) / mask.size
    
    if ratio > min_ratio:
        return tile_rgb
    else:
        return None

finish = 3072 * 1000000
svs_files = sorted(str(path) for path in data_root.rglob("*.svs"))
if not svs_files:
    raise RuntimeError(f"No SVS files found under {data_root}")

# Open the output file in write mode ('w')
# This will create the file if it doesn't exist or overwrite it if it does.
with open(output_filename, 'w') as f:
    print(f"Starting patch sampling. Output will be saved to {output_filename}")
    print("\nFor our OpenMidnight checkpoint we ran this script until we reached 29 million patches and then manually force-quit the script. You can adjust the 'finish' variable as needed.")
    
    for e in range(0, finish):
        for path in svs_files:
            image = OpenSlide(path)
            
            # Iterate through each level of the slide
            for level in range(0, image.level_count):
                
                # Get dimensions for the current level being processed
                height = image.level_dimensions[0][1]
                width = image.level_dimensions[0][0]
                
                # Ensure dimensions are valid for patch extraction
                if width < patch_size or height < patch_size:
                    continue

                tries = 0
                while True:
                    tries += 1
                    
                    # Randomly select a top-left coordinate for the patch
                    x = random.randint(0, width - patch_size)
                    y = random.randint(0, height - patch_size)
                    
                    # Read the region from the slide
                    patch = image.read_region((x, y), level=level, size=(patch_size, patch_size))
                    
                    # Check if the patch contains enough tissue
                    res = hsv(patch)
                    
                    if res is not None:
                        # If the patch is valid, write its info to the file
                        output_line = f"{path} {x} {y} {level}\n"
                        f.write(output_line)
                        break # Move to the next level/image
                
                    if tries >= max_tries:
                        # If 1000 random patches at this level are invalid, move on
                        break
            image.close()

# Shuffle the collected entries once generation finishes
with open(output_filename, 'r') as f:
    lines = f.readlines()

random.shuffle(lines)

with open(output_filename, 'w') as f:
    f.writelines(lines)

print("Done")
