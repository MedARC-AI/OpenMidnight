import os
import torch
import h5py
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from torchvision import transforms

from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup

def get_args():
    parser = argparse.ArgumentParser("OpenMidnight Feature Extractor")
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="tcga_embeddings.h5")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-patches", type=int, default=2000000)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()

class FastPatchedDataset(torch.utils.data.IterableDataset):
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        ds_sharded = self.ds.shard(num_shards=worker_info.num_workers, index=worker_info.id) if worker_info else self.ds

        for item in ds_sharded:
            try:
                img_array = np.frombuffer(item["image_bytes"], np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)

                tensor_img = self.transform(img_pil)
                meta = f"{item['slide_path']},{item['x']},{item['y']}"
                yield tensor_img, meta
            except Exception:
                continue

@torch.no_grad()
def extract_features(args):
    device = torch.device("cuda")

    print("Loading and compiling model...")
    cfg = setup(args)
    model, _ = build_model_from_cfg(cfg, only_teacher=True)

    state_dict = torch.load(args.checkpoint, map_location="cpu")["teacher"]
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    model = torch.compile(model)

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class FastPatchedDataset(torch.utils.data.IterableDataset):
        def __init__(self, ds, transform, max_patches):
            self.ds = ds
            self.transform = transform
            self.max_patches = max_patches

        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            ds_sharded = self.ds.shard(num_shards=worker_info.num_workers, index=worker_info.id) if worker_info else self.ds

            count = 0
            for item in ds_sharded:
                if count >= (self.max_patches // (worker_info.num_workers if worker_info else 1)): break


                img_array = np.frombuffer(item["image_bytes"], np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)

                yield self.transform(img), f"{item['slide_path']},{item['x']},{item['y']}"
                count += 1

    local_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.parquet')]
    raw_ds = load_dataset("parquet", data_files=local_files, split="train", streaming=True)

    loader = torch.utils.data.DataLoader(
        FastPatchedDataset(raw_ds, transform, args.max_patches),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )


    WRITE_BUFFER_SIZE = 10000
    embed_buffer = []
    meta_buffer = []

    embed_dim = model.embed_dim
    with h5py.File(args.output_file, 'w') as f:
        dset_embeds = f.create_dataset("embeddings", shape=(0, embed_dim), maxshape=(None, embed_dim), dtype='float32', chunks=(args.batch_size, embed_dim))
        dset_meta = f.create_dataset("metadata", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())

        pbar = tqdm(total=args.max_patches, desc="⚡ Extracting")

        for imgs, metas in loader:
            imgs = imgs.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = model(imgs, is_training=False).cpu().numpy()

            embed_buffer.append(features)
            meta_buffer.extend(metas)

            if len(meta_buffer) >= WRITE_BUFFER_SIZE:
                curr_len = len(dset_embeds)
                all_feats = np.concatenate(embed_buffer, axis=0)
                actual_write_size = len(all_feats)

                dset_embeds.resize(curr_len + actual_write_size, axis=0)
                dset_embeds[curr_len:] = all_feats

                dset_meta.resize(curr_len + actual_write_size, axis=0)
                dset_meta[curr_len:] = meta_buffer

                embed_buffer, meta_buffer = [], [] # Clear RAM

            pbar.update(len(features))

if __name__ == "__main__":
    extract_features(get_args())
