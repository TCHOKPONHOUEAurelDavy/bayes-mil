"""Generate a synthetic MIL dataset for a single MNIST-based task."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
import torch

from processing_scripts.mnist_number_datasets import DATASET_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MNIST-based MIL slides using the original task rules.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(DATASET_CLASSES.keys()),
        help="Which MNIST task to generate (e.g. mnist_fourbags).",
    )
    parser.add_argument("--mnist-root", type=Path, default=Path.home() / ".torch" / "datasets")
    parser.add_argument("--num-bags", type=int, default=120)
    parser.add_argument("--bag-size", type=int, default=12)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument(
        "--sampling",
        type=str,
        default="hierarchical",
        choices=["hierarchical", "uniform", "unique"],
        help="Sampling strategy for selecting digits within each bag.",
    )
    parser.add_argument("--slides-per-case", type=int, default=2)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_grid_coords(num_instances: int, patch_size: int = 28) -> np.ndarray:
    grid = int(math.ceil(math.sqrt(num_instances)))
    coords = []
    for idx in range(num_instances):
        row = idx // grid
        col = idx % grid
        coords.append((col * patch_size, row * patch_size))
    return np.asarray(coords, dtype=np.int32)


def write_h5(features: np.ndarray, coords: np.ndarray, destination: Path) -> None:
    ensure_dir(destination.parent)
    with h5py.File(destination, "w") as handle:
        handle.create_dataset("features", data=features)
        handle.create_dataset("coords", data=coords)


def append_shape(shape_file: Path, slide_id: str, coords: np.ndarray, patch_size: int = 28) -> None:
    width = int(coords[:, 0].max() + patch_size)
    height = int(coords[:, 1].max() + patch_size)
    with open(shape_file, "a", encoding="utf-8") as handle:
        handle.write(f"{slide_id},{width},{height}\n")


def stratified_folds(slides: List[Dict[str, str]], k_folds: int, seed: int) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    buckets: Dict[int, List[str]] = {}
    for slide in slides:
        buckets.setdefault(slide["label"], []).append(slide["slide_id"])
    folds = [[] for _ in range(k_folds)]
    for slide_ids in buckets.values():
        rng.shuffle(slide_ids)
        for idx, slide_id in enumerate(slide_ids):
            folds[idx % k_folds].append(slide_id)
    for fold in folds:
        fold.sort()
    return folds


def save_splits(
    slides: List[Dict[str, str]],
    task: str,
    k_folds: int,
    seed: int,
    output_dir: Path,
) -> None:
    folds = stratified_folds(slides, k_folds, seed)
    split_root = output_dir / "splits" / task
    ensure_dir(split_root)

    label_lookup = {slide["slide_id"]: slide["label"] for slide in slides}
    case_lookup = {slide["slide_id"]: slide["case_id"] for slide in slides}

    def pad(sequence: List[str], target: int) -> List[str]:
        return sequence + [""] * (target - len(sequence))

    max_len = max(len(fold) for fold in folds)

    for fold_idx in range(k_folds):
        test_ids = folds[fold_idx]
        val_ids = folds[(fold_idx + 1) % k_folds]
        train_ids: List[str] = []
        for other_idx, fold in enumerate(folds):
            if other_idx not in {fold_idx, (fold_idx + 1) % k_folds}:
                train_ids.extend(fold)
        train_ids.sort()

        wide = pd.DataFrame(
            {
                "train": pad(train_ids, max_len),
                "val": pad(val_ids, max_len),
                "test": pad(test_ids, max_len),
            }
        )
        wide.to_csv(split_root / f"splits_{fold_idx}.csv", index=False)

        bool_rows = []
        for slide_id in train_ids:
            bool_rows.append([slide_id, True, False, False])
        for slide_id in val_ids:
            bool_rows.append([slide_id, False, True, False])
        for slide_id in test_ids:
            bool_rows.append([slide_id, False, False, True])
        pd.DataFrame(bool_rows, columns=["slide_id", "train", "val", "test"]).to_csv(
            split_root / f"splits_{fold_idx}_bool.csv", index=False
        )

        descriptor_rows = []
        for slide_id in train_ids:
            descriptor_rows.append(
                {
                    "slide_id": slide_id,
                    "split": "train",
                    "label": label_lookup[slide_id],
                    "case_id": case_lookup[slide_id],
                }
            )
        for slide_id in val_ids:
            descriptor_rows.append(
                {
                    "slide_id": slide_id,
                    "split": "val",
                    "label": label_lookup[slide_id],
                    "case_id": case_lookup[slide_id],
                }
            )
        for slide_id in test_ids:
            descriptor_rows.append(
                {
                    "slide_id": slide_id,
                    "split": "test",
                    "label": label_lookup[slide_id],
                    "case_id": case_lookup[slide_id],
                }
            )
        pd.DataFrame(descriptor_rows).to_csv(
            split_root / f"splits_{fold_idx}_descriptor.csv", index=False
        )


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_cls = DATASET_CLASSES[args.dataset]
    dataset = dataset_cls(
        num_numbers=10,
        num_bags=args.num_bags,
        num_instances=args.bag_size,
        noise=args.noise,
        threshold=args.threshold,
        sampling=args.sampling,
        features_type="mnist_pixels",
        mnist_root=args.mnist_root,
        seed=args.seed,
    )

    output_dir = args.output_dir
    ensure_dir(output_dir)
    ensure_dir(output_dir / "h5_files")

    shape_file = output_dir / "images_shape.txt"
    if shape_file.exists():
        shape_file.unlink()

    csv_path = output_dir / f"{args.dataset}.csv"
    slides: List[Dict[str, str]] = []

    for index in range(len(dataset)):
        item = dataset[index]
        slide_id = f"slide_{index:04d}"
        case_id = f"case_{index // args.slides_per_case:04d}"

        features = item["features"].reshape(args.bag_size, -1).numpy()
        coords = make_grid_coords(args.bag_size)

        write_h5(features, coords, output_dir / "h5_files" / f"{slide_id}.h5")
        append_shape(shape_file, slide_id, coords)

        label = int(item["targets"].item())
        slides.append(
            {
                "case_id": case_id,
                "slide_id": slide_id,
                "label": label,
            }
        )

    pd.DataFrame(slides).to_csv(csv_path, index=False)

    save_splits(slides, args.dataset, args.k_folds, args.seed, output_dir)


if __name__ == "__main__":
    main()
