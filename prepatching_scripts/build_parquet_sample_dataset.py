#!/usr/bin/env python3
"""
Generate a Parquet dataset of TCGA patches described in sample_dataset_30.txt.

Each line in the input text file is expected to follow the format emitted by
create_sample_dataset_txt.py:

    /path/to/slide.svs <x> <y> <level>

Lines that include MPP metadata from the ablation sampler are also accepted:

    /path/to/slide.svs <x> <y> <level> <mpp_x> <mpp_y>

The script opens every referenced slide once, extracts a 224x224 RGB patch at the
requested coordinates/level, and writes the samples into per-task Parquet files.
It supports PNG/JPEG/raw outputs, task chunking, resume markers, and optional
multi-processing. Results land in /data/TCGA_parquet_sample30/ unless overridden.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import multiprocessing as mp
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
import os
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from openslide import OpenSlide


@dataclass(frozen=True)
class PatchSpec:
    slide_path: str
    x: int
    y: int
    level: int


@dataclass(frozen=True)
class TaskConfig:
    tile_size: int
    encoding: str
    max_open_slides: int
    output_dir: str
    progress_dir: str
    parquet_compression: str | None


CACHE_LIMIT_ENV = "PARQUET_MAX_OPEN_SLIDES"
try:
    _env_cache_limit = int(os.environ.get(CACHE_LIMIT_ENV, "16"))
except (TypeError, ValueError):
    _env_cache_limit = 16
DEFAULT_MAX_OPEN_SLIDES = max(1, _env_cache_limit)
MAX_OPEN_SLIDES = DEFAULT_MAX_OPEN_SLIDES
SLIDE_CACHE: "OrderedDict[str, OpenSlide]" = OrderedDict()


def prune_slide_cache() -> None:
    """Ensure the slide cache does not exceed MAX_OPEN_SLIDES."""
    if MAX_OPEN_SLIDES < 1:
        return
    while len(SLIDE_CACHE) > MAX_OPEN_SLIDES:
        old_path, old_slide = SLIDE_CACHE.popitem(last=False)
        try:
            old_slide.close()
        except Exception:
            logging.exception("Failed to close slide %s during cache prune.", old_path)


def set_max_open_slides(limit: int) -> None:
    """Update cache limit and prune if needed."""
    global MAX_OPEN_SLIDES
    limit = max(1, int(limit))
    if limit == MAX_OPEN_SLIDES:
        return
    MAX_OPEN_SLIDES = limit
    prune_slide_cache()


def ensure_cache_limit(limit: int) -> None:
    """Idempotent helper invoked inside workers."""
    try:
        set_max_open_slides(limit)
    except Exception:
        logging.exception("Unexpected failure updating slide cache limit.")


@atexit.register
def close_all_slides() -> None:
    """Close any slides left in the cache when the process exits."""
    while SLIDE_CACHE:
        path, slide = SLIDE_CACHE.popitem(last=False)
        try:
            slide.close()
        except Exception:
            logging.exception("Failed to close slide %s during interpreter shutdown.", path)


def get_slide(path: str) -> OpenSlide:
    slide = SLIDE_CACHE.get(path)
    if slide is None:
        slide = OpenSlide(path)
        SLIDE_CACHE[path] = slide
        SLIDE_CACHE.move_to_end(path, last=True)
        prune_slide_cache()
    else:
        SLIDE_CACHE.move_to_end(path, last=True)
    return slide


def parse_specs(spec_file: Path) -> List[PatchSpec]:
    specs: List[PatchSpec] = []
    with spec_file.open("r", encoding="utf-8") as fh:
        for idx, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 4:
                slide_path, x, y, level = parts
            elif len(parts) == 6:
                slide_path, x, y, level, *_mpp = parts
            else:
                raise ValueError(f"Invalid spec line {idx}: {line}")
            specs.append(PatchSpec(slide_path, int(x), int(y), int(level)))
    if not specs:
        raise ValueError(f"No patch specifications found in {spec_file}")
    logging.info("Parsed %d patch specifications from %s.", len(specs), spec_file)
    return specs


def group_by_slide(specs: Sequence[PatchSpec]) -> List[Tuple[str, List[PatchSpec]]]:
    """Group patch specs so each slide is opened only once."""
    grouped: "OrderedDict[str, List[PatchSpec]]" = OrderedDict()
    for spec in specs:
        grouped.setdefault(spec.slide_path, []).append(spec)
    return [(slide, patches) for slide, patches in grouped.items()]


def build_tasks(
    groups: Sequence[Tuple[str, Sequence[PatchSpec]]],
    max_patches_per_task: int,
    *,
    start_index: int = 0,
) -> List[Tuple[str, List[PatchSpec], str]]:
    """Chunk grouped specs to improve parallel balance on large nodes."""
    tasks: List[Tuple[str, List[PatchSpec], str]] = []
    idx = start_index
    for slide, patches in groups:
        chunk = len(patches) if max_patches_per_task <= 0 else max_patches_per_task
        for start in range(0, len(patches), chunk):
            chunk_specs = list(patches[start : start + chunk])
            task_id = f"{idx:08d}"
            tasks.append((slide, chunk_specs, task_id))
            idx += 1
    return tasks


def find_next_task_index(output_dir: Path) -> int:
    """Return the next sequential task index based on existing Parquet files."""
    max_idx = -1
    for candidate in output_dir.glob("*.parquet"):
        stem = candidate.stem
        if stem.isdigit():
            try:
                max_idx = max(max_idx, int(stem))
            except ValueError:
                continue
    return max_idx + 1


def load_completed_task_ids(progress_dir: Path, output_dir: Path) -> Set[str]:
    """Load completed task IDs from progress markers, pruning stale entries."""
    done_ids: Set[str] = set()
    for marker in progress_dir.iterdir():
        if not marker.is_file():
            continue
        if marker.name == "task_offset.txt":
            continue
        parquet_path = output_dir / f"{marker.stem}.parquet"
        if parquet_path.exists():
            done_ids.add(marker.stem)
        else:
            marker.unlink()
    return done_ids


def process_task_worker(
    task: Tuple[str, Sequence[PatchSpec], str],
    config: TaskConfig,
) -> str:
    """Process a single task and materialize it to a Parquet file."""
    slide_path, specs, task_id = task
    ensure_cache_limit(config.max_open_slides)
    slide = get_slide(slide_path)
    buf = BytesIO() if config.encoding in ("png", "jpeg") else None

    task_ids: List[str] = []
    slide_paths: List[str] = []
    xs: List[int] = []
    ys: List[int] = []
    levels: List[int] = []
    tile_sizes: List[int] = []
    level_downsamples: List[float] = []
    image_bytes: List[bytes] = []
    image_dtypes: List[str] = []

    for spec in specs:
        if spec.level < 0 or spec.level >= slide.level_count:
            raise ValueError(f"Level {spec.level} invalid for {slide_path}")
        region = slide.read_region(
            (spec.x, spec.y), spec.level, (config.tile_size, config.tile_size)
        ).convert("RGB")

        task_ids.append(task_id)
        slide_paths.append(slide_path)
        xs.append(spec.x)
        ys.append(spec.y)
        levels.append(spec.level)
        tile_sizes.append(config.tile_size)
        level_downsamples.append(float(slide.level_downsamples[spec.level]))

        if config.encoding == "png":
            buf.seek(0)
            buf.truncate(0)
            region.save(buf, format="PNG", optimize=True)
            image_bytes.append(buf.getvalue())
            image_dtypes.append("uint8")
        elif config.encoding == "jpeg":
            buf.seek(0)
            buf.truncate(0)
            region.save(buf, format="JPEG", quality=95, optimize=True)
            image_bytes.append(buf.getvalue())
            image_dtypes.append("uint8")
        else:
            arr = np.asarray(region, dtype=np.uint8)
            image_bytes.append(arr.tobytes())
            image_dtypes.append(str(arr.dtype))

    table = pa.table(
        {
            "task_id": pa.array(task_ids, type=pa.string()),
            "slide_path": pa.array(slide_paths, type=pa.string()),
            "x": pa.array(xs, type=pa.int32()),
            "y": pa.array(ys, type=pa.int32()),
            "level": pa.array(levels, type=pa.int32()),
            "tile_size": pa.array(tile_sizes, type=pa.int32()),
            "level_downsample": pa.array(level_downsamples, type=pa.float32()),
            "image_dtype": pa.array(image_dtypes, type=pa.string()),
            "image_bytes": pa.array(image_bytes, type=pa.binary()),
        }
    )

    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"image_encoding": config.encoding.encode("utf-8"),
            b"tile_size": str(config.tile_size).encode("utf-8"),
        }
    )
    table = table.replace_schema_metadata(metadata)

    output_path = Path(config.output_dir) / f"{task_id}.parquet"
    pq.write_table(table, output_path, compression=config.parquet_compression)
    progress_path = Path(config.progress_dir) / f"{task_id}.done"
    progress_path.write_text("1\n")
    return task_id


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Parquet dataset of TCGA patches listed in a text file."
    )
    parser.add_argument(
        "--spec-file",
        type=Path,
        default=Path("sample_dataset_30.txt"),
        help="Path to the text file produced by create_sample_dataset_txt.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/TCGA_parquet_sample30/"),
        help="Directory where Parquet files will be written (default: /data/TCGA_parquet_sample30/).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=224,
        help="Square tile size (pixels) to read from each slide.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker processes for parallel patch extraction (default: auto based on CPUs).",
    )
    parser.add_argument(
        "--start-method",
        choices=("fork", "spawn", "forkserver"),
        default=None,
        help="Multiprocessing start method used by the worker pool (default: library default).",
    )
    parser.add_argument(
        "--mode",
        choices=("overwrite", "append"),
        default="overwrite",
        help="Whether to overwrite or append to an existing Parquet dataset.",
    )
    parser.add_argument(
        "--keep-order",
        action="store_true",
        help="Keep samples in the same order as listed in the spec file.",
    )
    parser.add_argument(
        "--encoding",
        choices=("png", "jpeg", "raw"),
        default="png",
        help="Patch serialization format (png: smaller, jpeg/raw: faster).",
    )
    parser.add_argument(
        "--task-chunk-size",
        type=int,
        default=42000,
        help="Max patches per task to balance multi-process workloads (<=0 disables chunking). This is an arbitrarily high number which basically means that all patches for a slide will be processed in a single task. Can consider letting tasks span multiple slides in a future code revision if you want parquet files to be larger size.",
    )
    parser.add_argument(
        "--shuffle-tasks",
        action="store_true",
        help="Shuffle task order (ignored when --keep-order is set).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks that completed in a previous run using progress markers.",
    )
    parser.add_argument(
        "--max-open-slides",
        type=int,
        default=DEFAULT_MAX_OPEN_SLIDES,
        help=(
            "Maximum slides cached per worker before least-recently-used eviction; "
        ),
    )
    parser.add_argument(
        "--parquet-compression",
        default="none",
        choices=("none", "snappy", "gzip", "brotli", "zstd", "lz4"),
        help="Compression codec for Parquet output (default: none).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    ensure_cache_limit(args.max_open_slides)

    if not args.spec_file.exists():
        raise SystemExit(f"Spec file not found: {args.spec_file}")

    specs = parse_specs(args.spec_file)
    if not specs:
        raise SystemExit("No valid patch specs to process. Aborting.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    compression = None if args.parquet_compression == "none" else args.parquet_compression

    existing_parquet = [
        path for path in args.output_dir.glob("*.parquet") if path.is_file()
    ]
    if args.mode == "overwrite" and not args.resume:
        for path in existing_parquet:
            try:
                path.unlink()
            except Exception:
                logging.exception("Failed to remove %s during overwrite preparation.", path)
                raise

    progress_dir = args.output_dir / ".resume"
    progress_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for marker in progress_dir.iterdir():
            if marker.is_file():
                marker.unlink()

    done_ids = load_completed_task_ids(progress_dir, args.output_dir)
    offset_file = progress_dir / "task_offset.txt"

    if args.resume:
        if offset_file.exists():
            try:
                start_index = int(offset_file.read_text().strip())
            except ValueError:
                logging.warning("Ignoring malformed task offset metadata in %s.", offset_file)
                start_index = 0
        elif done_ids:
            try:
                start_index = min(int(task_id) for task_id in done_ids)
            except ValueError:
                start_index = 0
        elif args.mode == "append":
            start_index = find_next_task_index(args.output_dir)
        else:
            start_index = 0
    else:
        start_index = find_next_task_index(args.output_dir) if args.mode == "append" else 0

    try:
        offset_file.write_text(f"{start_index}\n")
    except Exception:
        logging.exception("Failed to persist task offset metadata to %s.", offset_file)
        raise

    grouped = group_by_slide(specs)
    tasks = build_tasks(
        grouped,
        args.task_chunk_size,
        start_index=start_index,
    )

    if done_ids:
        tasks = [task for task in tasks if task[2] not in done_ids]

    if args.shuffle_tasks and not args.keep_order:
        random.shuffle(tasks)

    if not tasks:
        logging.info("No tasks remaining after resume check.")
        return

    pending_patches = sum(len(task[1]) for task in tasks)
    cpu_count = os.cpu_count() or 1
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = max(1, min(cpu_count, 64))

    logging.info(
        "Building Parquet dataset in %s (num_workers=%d, mode=%s, encoding=%s, compression=%s).",
        args.output_dir,
        num_workers,
        args.mode,
        args.encoding,
        compression or "none",
    )
    logging.info(
        "Prepared %d task(s) across %d slide(s) [chunk_size=%d, pending_patches=%d].",
        len(tasks),
        len(grouped),
        args.task_chunk_size,
        pending_patches,
    )
    if done_ids:
        logging.info("Skipping %d previously completed task(s).", len(done_ids))

    config = TaskConfig(
        tile_size=args.tile_size,
        encoding=args.encoding,
        max_open_slides=args.max_open_slides,
        output_dir=str(args.output_dir),
        progress_dir=str(progress_dir),
        parquet_compression=compression,
    )

    if num_workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            task_id = process_task_worker(task, config)
            logging.info(
                "Completed task %s (%d/%d) for slide %s (%d patch(es)).",
                task_id,
                idx,
                len(tasks),
                task[0],
                len(task[1]),
            )
    else:
        mp_context = mp.get_context(args.start_method) if args.start_method else None
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
            futures = {executor.submit(process_task_worker, task, config): task for task in tasks}
            for idx, future in enumerate(as_completed(futures), start=1):
                slide_path, specs_for_task, task_id_hint = futures[future]
                try:
                    completed_task_id = future.result()
                except Exception:
                    logging.exception(
                        "Task %s failed for slide %s (%d patch(es)).",
                        task_id_hint,
                        slide_path,
                        len(specs_for_task),
                    )
                    raise
                logging.info(
                    "Completed task %s (%d/%d) for slide %s (%d patch(es)).",
                    completed_task_id,
                    idx,
                    len(tasks),
                    slide_path,
                    len(specs_for_task),
                )
    logging.info("Dataset build complete.")


if __name__ == "__main__":
    main()

# python3 build_parquet_sample_dataset.py \
#   --spec-file /data/TCGA/sample_dataset_ablation.txt \
#   --output-dir /data/TCGA_ablations_baseline/ \
#   --encoding jpeg \
#   --shuffle-tasks \
#   --num-workers 32 \
#   --max-open-slides 1 \
#   --start-method spawn \
#   --mode append \
#   --resume


# PS: Always set max-open-slides to 1 because you'll go out of memory otherwise;
