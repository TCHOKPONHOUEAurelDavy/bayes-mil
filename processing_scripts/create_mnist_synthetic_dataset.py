"""Generate a synthetic MIL dataset from MNIST digits for a single task.

The script keeps the output layout compatible with Bayes-MIL while keeping the
implementation intentionally small: each interpretability task is built through a
simple helper that crafts digit sequences for the requested class. This makes it
straightforward to verify that every dataset respects the original rules.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from processing_scripts.mnist_interpretability_tasks import TASK_METADATA_FNS

TASK_LABEL_MAPS = {
    "mnist_fourbags": {0: "none", 1: "mostly_eight", 2: "mostly_nine", 3: "both"},
    "mnist_even_odd": {0: "odd_majority", 1: "even_majority"},
    "mnist_adjacent_pairs": {0: "no_adjacent_pairs", 1: "has_adjacent_pairs"},
    "mnist_fourbags_plus": {
        0: "none",
        1: "three_five",
        2: "one_only",
        3: "one_and_seven",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Create a synthetic MNIST MIL dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=sorted(TASK_LABEL_MAPS.keys()),
        help="Name of the MNIST interpretability task to generate",
    )
    parser.add_argument("--mnist-root", type=Path, default=Path.home() / ".torch" / "datasets")
    parser.add_argument("--num-slides", type=int, default=120)
    parser.add_argument("--min-patches", type=int, default=6)
    parser.add_argument("--max-patches", type=int, default=18)
    parser.add_argument("--slides-per-case", type=int, default=2)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--minority-fraction",
        type=float,
        default=0.25,
        help="Ensure every label appears in at least this fraction of the dataset",
    )
    return parser.parse_args()


def load_mnist_images(root: Path) -> tuple[np.ndarray, np.ndarray]:
    dataset = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    images = dataset.data.numpy().astype(np.float32) / 255.0
    labels = dataset.targets.numpy().astype(np.int64)
    return images, labels


def index_digits(labels: np.ndarray) -> Dict[int, List[int]]:
    return {digit: np.where(labels == digit)[0].tolist() for digit in range(10)}


def draw_many(rng: random.Random, choices: Sequence[int], count: int) -> List[int]:
    return [choices[rng.randrange(len(choices))] for _ in range(count)]


def digits_for_label(task: str, label: int, bag_size: int, rng: random.Random) -> List[int]:
    if task == "mnist_fourbags":
        others = [d for d in range(10) if d not in (8, 9)]
        if label == 0:
            digits = draw_many(rng, others, bag_size)
        elif label == 1:
            bag_size = max(bag_size, 1)
            digits = [8] + draw_many(rng, others, bag_size - 1)
        elif label == 2:
            bag_size = max(bag_size, 1)
            digits = [9] + draw_many(rng, others, bag_size - 1)
        else:
            bag_size = max(bag_size, 2)
            digits = [8, 9] + draw_many(rng, others, bag_size - 2)
    elif task == "mnist_even_odd":
        evens = [0, 2, 4, 6, 8]
        odds = [1, 3, 5, 7, 9]
        half = bag_size // 2
        if label == 1:
            num_even = half + 1
            num_odd = bag_size - num_even
            digits = draw_many(rng, evens, num_even) + draw_many(rng, odds, num_odd)
        else:
            num_odd = half + 1
            num_even = bag_size - num_odd
            digits = draw_many(rng, odds, num_odd) + draw_many(rng, evens, num_even)
    elif task == "mnist_adjacent_pairs":
        if label == 1:
            pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
            first, second = pairs[rng.randrange(len(pairs))]
            bag_size = max(bag_size, 2)
            digits = [first, second] + draw_many(rng, list(range(10)), bag_size - 2)
        else:
            safe_digits = [0, 2, 4, 6, 7, 8, 9]
            digits = draw_many(rng, safe_digits, bag_size)
    else:  # mnist_fourbags_plus
        fillers = [0, 2, 4, 6, 8, 9]
        if label == 1:
            bag_size = max(bag_size, 2)
            digits = [3, 5] + draw_many(rng, fillers + [1], bag_size - 2)
        elif label == 2:
            bag_size = max(bag_size, 1)
            digits = [1] + draw_many(rng, fillers, bag_size - 1)
        elif label == 3:
            bag_size = max(bag_size, 2)
            digits = [1, 7] + draw_many(rng, fillers, bag_size - 2)
        else:
            digits = draw_many(rng, fillers, bag_size)
    rng.shuffle(digits)
    return digits


def sample_features(
    digits: Sequence[int],
    rng: random.Random,
    images: np.ndarray,
    digit_to_indices: Dict[int, Sequence[int]],
) -> np.ndarray:
    indices = [rng.choice(digit_to_indices[digit]) for digit in digits]
    patches = images[indices]
    return patches.reshape(len(digits), -1).astype(np.float32)


def make_grid_coords(num_instances: int, patch_size: int = 28) -> tuple[np.ndarray, int, int]:
    grid = int(math.ceil(math.sqrt(num_instances)))
    coords = []
    for idx in range(num_instances):
        row = idx // grid
        col = idx % grid
        coords.append((col * patch_size, row * patch_size))
    width = grid * patch_size
    height = grid * patch_size
    return np.asarray(coords, dtype=np.int32), width, height


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plan_label_counts(task: str, num_slides: int, minority_fraction: float) -> Dict[int, int]:
    labels = sorted(TASK_LABEL_MAPS[task].keys())
    min_per_label = max(1, math.ceil(num_slides * minority_fraction))
    counts = {label: min_per_label for label in labels}
    total = sum(counts.values())
    if total < num_slides:
        remaining = num_slides - total
        index = 0
        while remaining > 0:
            counts[labels[index % len(labels)]] += 1
            index += 1
            remaining -= 1
    return counts


def write_h5(features: np.ndarray, coords: np.ndarray, destination: Path) -> None:
    ensure_dir(destination.parent)
    with h5py.File(destination, "w") as handle:
        handle.create_dataset("features", data=features)
        handle.create_dataset("coords", data=coords)


def append_shape(shape_file: Path, slide_id: str, width: int, height: int) -> None:
    with open(shape_file, "a", encoding="utf-8") as handle:
        handle.write(f"{slide_id},{width},{height}\n")


def bundle_to_dict(task: str, numbers: Sequence[int]) -> dict:
    numbers_tensor = torch.tensor(numbers, dtype=torch.long)
    bundle = TASK_METADATA_FNS[task](numbers_tensor)
    label_map = TASK_LABEL_MAPS[task]
    label_name = label_map[bundle.target]
    evidence = {k: v.detach().cpu().tolist() for k, v in bundle.evidence.items()}
    if bundle.instance_labels is None:
        instance_labels = None
    else:
        instance_labels = bundle.instance_labels.detach().cpu().tolist()
    return {
        "numbers": list(numbers),
        "target": int(bundle.target),
        "label": label_name,
        "evidence": evidence,
        "instance_labels": instance_labels,
    }


def stratified_folds(slides: Sequence[dict], k_folds: int, seed: int) -> List[List[str]]:
    buckets: Dict[str, List[str]] = {}
    rng = random.Random(seed)
    for slide in slides:
        buckets.setdefault(slide["label"], []).append(slide["slide_id"])
    for ids in buckets.values():
        rng.shuffle(ids)
    folds = [[] for _ in range(k_folds)]
    for ids in buckets.values():
        for idx, slide_id in enumerate(ids):
            folds[idx % k_folds].append(slide_id)
    for fold in folds:
        fold.sort()
    return folds


def to_wide(train: Sequence[str], val: Sequence[str], test: Sequence[str]) -> pd.DataFrame:
    max_len = max(len(train), len(val), len(test))
    pad = lambda seq: list(seq) + [""] * (max_len - len(seq))
    return pd.DataFrame({"train": pad(train), "val": pad(val), "test": pad(test)})


def to_boolean(train: Sequence[str], val: Sequence[str], test: Sequence[str]) -> pd.DataFrame:
    rows = [[slide_id, True, False, False] for slide_id in train]
    rows += [[slide_id, False, True, False] for slide_id in val]
    rows += [[slide_id, False, False, True] for slide_id in test]
    return pd.DataFrame(rows, columns=["slide_id", "train", "val", "test"])


def save_splits(slides: Sequence[dict], task: str, k_folds: int, seed: int, output_dir: Path) -> None:
    folds = stratified_folds(slides, k_folds, seed)
    case_lookup = {slide["slide_id"]: slide["case_id"] for slide in slides}
    label_lookup = {slide["slide_id"]: slide["label"] for slide in slides}
    split_root = output_dir / "splits" / task
    ensure_dir(split_root)
    for fold_idx in range(k_folds):
        test_ids = folds[fold_idx]
        val_ids = folds[(fold_idx + 1) % k_folds]
        train_ids = sorted(
            slide_id
            for other_idx, fold in enumerate(folds)
            if other_idx not in {fold_idx, (fold_idx + 1) % k_folds}
            for slide_id in fold
        )
        to_wide(train_ids, val_ids, test_ids).to_csv(
            split_root / f"splits_{fold_idx}.csv", index=False
        )
        to_boolean(train_ids, val_ids, test_ids).to_csv(
            split_root / f"splits_{fold_idx}_bool.csv", index=False
        )
        descriptor_rows = []
        for split_name, ids in ("train", train_ids), ("val", val_ids), ("test", test_ids):
            for slide_id in ids:
                descriptor_rows.append(
                    {
                        "slide_id": slide_id,
                        "case_id": case_lookup[slide_id],
                        "label": label_lookup[slide_id],
                        "split": split_name,
                    }
                )
        pd.DataFrame(descriptor_rows).to_csv(
            split_root / f"splits_{fold_idx}_descriptor.csv", index=False
        )


def main() -> None:
    args = parse_args()
    if not (0.0 < args.minority_fraction <= 0.5):
        raise ValueError("--minority-fraction must be in (0, 0.5]")
    if args.min_patches > args.max_patches:
        raise ValueError("--min-patches cannot exceed --max-patches")

    rng = random.Random(args.seed)
    images, labels = load_mnist_images(args.mnist_root)
    digit_to_indices = index_digits(labels)

    output_dir = args.output_dir
    ensure_dir(output_dir)
    ensure_dir(output_dir / "h5_files")
    evidence_dir = output_dir / "evidence" / args.task
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    ensure_dir(evidence_dir)

    csv_path = output_dir / f"{args.task}.csv"
    if csv_path.exists():
        csv_path.unlink()

    shape_file = output_dir / "images_shape.txt"
    if shape_file.exists():
        shape_file.unlink()

    label_plan = plan_label_counts(args.task, args.num_slides, args.minority_fraction)
    slides: List[dict] = []

    slide_index = 0
    for label, target_count in sorted(label_plan.items()):
        for _ in range(target_count):
            bag_size = rng.randint(args.min_patches, args.max_patches)
            digits = digits_for_label(args.task, label, bag_size, rng)
            features = sample_features(digits, rng, images, digit_to_indices)
            coords, width, height = make_grid_coords(len(digits))

            slide_id = f"slide_{slide_index:04d}"
            case_id = f"case_{slide_index // args.slides_per_case:04d}"
            slide_index += 1

            write_h5(features, coords, output_dir / "h5_files" / f"{slide_id}.h5")
            append_shape(shape_file, slide_id, width, height)

            evidence_payload = bundle_to_dict(args.task, digits)
            torch.save(evidence_payload, evidence_dir / f"{slide_id}.pt")

            slides.append(
                {
                    "case_id": case_id,
                    "slide_id": slide_id,
                    "label": evidence_payload["label"],
                    "numbers": digits,
                }
            )

    df = pd.DataFrame(
        {
            "case_id": [slide["case_id"] for slide in slides],
            "slide_id": [slide["slide_id"] for slide in slides],
            "label": [slide["label"] for slide in slides],
        }
    )
    df.to_csv(csv_path, index=False)

    save_splits(slides, args.task, args.k_folds, args.seed, output_dir)


if __name__ == "__main__":
    main()
