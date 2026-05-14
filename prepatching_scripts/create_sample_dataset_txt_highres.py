import cv2
import random
from pathlib import Path
from openslide import OpenSlide
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="/block/TCGA")
parser.add_argument("--output", type=str, default="sample_dataset_highres_2M.txt")
parser.add_argument("--target_patches", type=int, default=1_000_000)
parser.add_argument("--max_tries", type=int, default=1000)
parser.add_argument("--patches_per_mag", type=int, default=5)
parser.add_argument("--workers", type=int, default=16)
args = parser.parse_args()

TARGET_MPPS = [1.0, 0.5, 0.25, 0.125]
TILE_SIZE = 448
MAX_UPSCALE = 2.0
MPP_KEY = "openslide.mpp-x"


def hsv_tissue_check(tile_rgb):
    tile = np.array(tile_rgb)
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(tile, np.array([90, 8, 103]), np.array([180, 255, 255]))
    return np.count_nonzero(mask) / mask.size > 0.6


def process_slide(task):
    path, seed, patches_per_mag, max_tries = task
    random.seed(seed)
    results = []

    try:
        slide = OpenSlide(path)
    except Exception:
        return results

    if MPP_KEY not in slide.properties:
        slide.close()
        return results

    native_mpp = float(slide.properties[MPP_KEY])
    lv0_w, lv0_h = slide.level_dimensions[0]

    for target_mpp in TARGET_MPPS:
        if native_mpp > MAX_UPSCALE * target_mpp:
            continue

        target_ds = target_mpp / native_mpp
        best_level = 0
        for l in range(slide.level_count):
            if slide.level_downsamples[l] <= target_ds:
                best_level = l

        level_mpp = native_mpp * slide.level_downsamples[best_level]
        read_size = int(round(TILE_SIZE * target_mpp / level_mpp))

        physical_lv0 = int(read_size * slide.level_downsamples[best_level])
        max_x = lv0_w - physical_lv0
        max_y = lv0_h - physical_lv0
        if max_x <= 0 or max_y <= 0:
            continue

        collected = 0
        tries = 0
        while collected < patches_per_mag and tries < max_tries:
            tries += 1
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            patch = slide.read_region((x, y), level=best_level, size=(read_size, read_size)).convert("RGB")
            if hsv_tissue_check(patch):
                results.append(f"{path} {x} {y} {best_level} {read_size}\n")
                collected += 1

    slide.close()
    return results


data_root = Path(args.data_root)
svs_files = sorted(str(p) for p in data_root.rglob("*.svs"))
if not svs_files:
    raise RuntimeError(f"No SVS files found under {data_root}")

print(f"Found {len(svs_files)} SVS files")
print(f"Target magnifications (um/px): {TARGET_MPPS}")
print(f"Tile size: {TILE_SIZE}px, max upscale: {MAX_UPSCALE}x")
print(f"Target: {args.target_patches} patches, workers: {args.workers}")

total = 0
pass_idx = 0
pbar = tqdm(total=args.target_patches, desc="Patches")
with open(args.output, 'w') as f:
    while total < args.target_patches:
        tasks = [(path, pass_idx * 100000 + i, args.patches_per_mag, args.max_tries)
                 for i, path in enumerate(svs_files)]
        random.shuffle(tasks)

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for results in executor.map(process_slide, tasks):
                for line in results:
                    if total >= args.target_patches:
                        break
                    f.write(line)
                    total += 1
                    pbar.update(1)
                if total >= args.target_patches:
                    break

        pass_idx += 1
        print(f"Pass {pass_idx} complete, {total} patches so far")

pbar.close()
print(f"Generated {total} patches. Shuffling...")

with open(args.output, 'r') as f:
    lines = f.readlines()

random.shuffle(lines)

with open(args.output, 'w') as f:
    f.writelines(lines)

print("Done")
