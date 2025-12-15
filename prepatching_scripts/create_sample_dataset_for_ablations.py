import cv2
import random
from pathlib import Path
from openslide import OpenSlide
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

data_root = Path("/data/TCGA")
output_filename = "/data/TCGA/sample_dataset_ablation.txt"
patch_size = 224
max_tries_per_level = 1000
max_patches = 500_000
patches_per_level = 100
seed = 0
workers = 10
MPP_X_KEY = "openslide.mpp-x"
MPP_Y_KEY = "openslide.mpp-y"

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

random.seed(seed)
svs_files = sorted(str(path) for path in data_root.rglob("*.svs"))
random.shuffle(svs_files)
if not svs_files:
    raise RuntimeError(f"No SVS files found under {data_root}")

def sample_slide(args):
    path, slide_idx, pass_idx = args
    random.seed(seed + pass_idx * 10_000 + slide_idx)
    image = OpenSlide(path)
    collected_lines = []

    props = image.properties
    if MPP_X_KEY not in props or MPP_Y_KEY not in props:
        image.close()
        print(f"Skipping slide without MPP metadata: {path}")
        return []

    base_mpp_x = float(props[MPP_X_KEY])
    base_mpp_y = float(props[MPP_Y_KEY])

    for level in range(0, image.level_count):
        height = image.level_dimensions[0][1]
        width = image.level_dimensions[0][0]
        if width < patch_size or height < patch_size:
            continue

        target_for_level = patches_per_level
        collected = 0
        tries = 0
        downsample = image.level_downsamples[level]
        mpp_x = base_mpp_x * downsample
        mpp_y = base_mpp_y * downsample
        while collected < target_for_level and tries < max_tries_per_level:
            tries += 1
            x = random.randint(0, width - patch_size)
            y = random.randint(0, height - patch_size)
            patch = image.read_region((x, y), level=level, size=(patch_size, patch_size))
            res = hsv(patch)
            if res is not None:
                collected_lines.append(f"{path} {x} {y} {level} {mpp_x} {mpp_y}\n")
                collected += 1
    image.close()
    return collected_lines

# Open the output file in write mode ('w')
# This will create the file if it doesn't exist or overwrite it if it does.
with open(output_filename, 'w') as f:
    print(f"Starting patch sampling (target: {max_patches} patches). Output will be saved to {output_filename}")
    
    patches_written = 0
    progress = tqdm(total=max_patches, desc="Patches collected")
    pass_idx = 0
    while patches_written < max_patches:
        patches_before = patches_written
        with ProcessPoolExecutor(max_workers=workers) as executor:
            tasks = ((path, idx, pass_idx) for idx, path in enumerate(svs_files))
            for lines in executor.map(sample_slide, tasks):
                for line in lines:
                    if patches_written >= max_patches:
                        break
                    f.write(line)
                    patches_written += 1
                    progress.update(1)
                if patches_written >= max_patches:
                    break
        pass_idx += 1
        if patches_written == patches_before:
            break
    progress.close()

# Shuffle the collected entries once generation finishes
with open(output_filename, 'r') as f:
    lines = f.readlines()

random.shuffle(lines)

with open(output_filename, 'w') as f:
    f.writelines(lines)

print("Done")
