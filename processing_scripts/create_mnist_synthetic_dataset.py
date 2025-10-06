"""Utility to generate synthetic MIL datasets from MNIST.

The resulting directory mimics the structure expected by ``Generic_MIL_Dataset``:

* ``h5_files`` contains one HDF5 file per synthetic "slide".
* ``mnist_binary.csv`` stores slide-level labels for a binary task.
* ``mnist_ternary.csv`` stores slide-level labels for a three-class task.
* ``splits/<task>/`` holds cross-validation splits aligned with ``main.py``.
* ``images_shape.txt`` records the spatial canvas size for every slide.

Example
-------

.. code-block:: bash

    python processing_scripts/create_mnist_synthetic_dataset.py \
        --output-dir data/mnist_mil --num-slides 150 --k-folds 5

The command above creates a dataset that can be consumed by the training
entry-point with ``--task mnist_binary`` or ``--task mnist_ternary``.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from torchvision import datasets, transforms


GROUP_MAPPING = {
    "low_digit": {0, 1, 2, 3},
    "mid_digit": {4, 5, 6},
    "high_digit": {7, 8, 9},
}


@dataclass(frozen=True)
class SlideExample:
    """Container for the metadata associated with one synthetic slide."""

    case_id: str
    slide_id: str
    binary_label: str
    ternary_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic MIL datasets using MNIST digits."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the synthetic dataset will be written.",
    )
    parser.add_argument(
        "--mnist-root",
        type=str,
        default=os.path.join(os.path.expanduser("~"), ".torch", "datasets"),
        help="Root directory used by torchvision to cache MNIST (default: %(default)s).",
    )
    parser.add_argument(
        "--num-slides",
        type=int,
        default=120,
        help="Total number of synthetic slides to create (default: %(default)s).",
    )
    parser.add_argument(
        "--min-patches",
        type=int,
        default=6,
        help="Minimum number of MNIST digits sampled per slide (default: %(default)s).",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=18,
        help="Maximum number of MNIST digits sampled per slide (default: %(default)s).",
    )
    parser.add_argument(
        "--slides-per-case",
        type=int,
        default=2,
        help="Number of slides grouped under the same synthetic case identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds emitted under the splits directory (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for reproducibility (default: %(default)s).",
    )

    return parser.parse_args()


def load_mnist_images(root: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the MNIST training set and return (images, labels).

    Images are converted to float32 and normalized to [0, 1].
    """

    dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    images = dataset.data.numpy().astype(np.float32) / 255.0
    labels = dataset.targets.numpy().astype(np.int64)
    return images, labels


def sample_slide_contents(
    rng: random.Random,
    digit_sequence: Sequence[int],
    image_pool: np.ndarray,
    digit_to_indices: Dict[int, Sequence[int]],
) -> Tuple[np.ndarray, List[int]]:
    """Sample MNIST digits following ``digit_sequence``."""

    chosen_indices = [rng.choice(digit_to_indices[digit]) for digit in digit_sequence]
    selected_images = image_pool[chosen_indices]
    features = selected_images.reshape(len(digit_sequence), -1)
    return features, list(digit_sequence)


def make_grid_coords(num_instances: int, patch_size: int = 28) -> Tuple[np.ndarray, int, int]:
    """Arrange patches on a square grid and return coordinates and canvas size."""

    grid_size = int(math.ceil(math.sqrt(num_instances)))
    coords = []
    for index in range(num_instances):
        row = index // grid_size
        col = index % grid_size
        coords.append((col * patch_size, row * patch_size))
    width = grid_size * patch_size
    height = grid_size * patch_size
    return np.array(coords, dtype=np.int32), width, height


def determine_labels(digits: Sequence[int]) -> Tuple[str, str]:
    """Assign binary and ternary slide labels based on contained digits."""

    binary_label = "positive" if any(digit >= 5 for digit in digits) else "negative"

    counts = {
        group: sum(1 for digit in digits if digit in members)
        for group, members in GROUP_MAPPING.items()
    }
    # Resolve ties deterministically by sorting on (count, group_name).
    ternary_label = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    return binary_label, ternary_label


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def majority_count(num_instances: int) -> int:
    """Return the minimal count required to secure a strict majority."""

    return max(1, num_instances // 2 + 1)


def generate_digit_sequence(
    rng: random.Random,
    bag_size: int,
    binary_label: str,
    ternary_label: str,
) -> Sequence[int]:
    """Create a digit sequence that realizes the requested labels."""

    if binary_label not in {"positive", "negative"}:
        raise ValueError(f"Unknown binary label: {binary_label}")
    if ternary_label not in GROUP_MAPPING:
        raise ValueError(f"Unknown ternary label: {ternary_label}")

    if binary_label == "negative" and ternary_label == "high_digit":
        raise ValueError("Cannot construct a negative slide with a high_digit majority.")

    digits: List[int] = []

    if binary_label == "negative":
        if ternary_label == "low_digit":
            digits = [rng.choice(tuple(GROUP_MAPPING["low_digit"])) for _ in range(bag_size)]
        else:  # ternary_label == "mid_digit"
            # Ensure a strict majority of "4" digits while allowing some lower digits.
            majority = majority_count(bag_size)
            digits = [4] * majority
            remaining = bag_size - majority
            pool = tuple(GROUP_MAPPING["low_digit"])
            digits.extend(rng.choice(pool) for _ in range(remaining))
            rng.shuffle(digits)
    else:  # binary positive
        if ternary_label == "low_digit":
            if bag_size == 1:
                raise ValueError(
                    "Cannot synthesize a positive slide with a single low_digit instance."
                )
            low_pool = tuple(GROUP_MAPPING["low_digit"])
            digits = [rng.choice(low_pool) for _ in range(bag_size)]
            # Inject a high-digit instance to flip the binary label while preserving majority.
            digits[-1] = rng.choice(tuple(GROUP_MAPPING["high_digit"]))
            rng.shuffle(digits)
        elif ternary_label == "mid_digit":
            mid_pool = tuple(GROUP_MAPPING["mid_digit"])
            digits = [rng.choice(mid_pool) for _ in range(majority_count(bag_size))]
            if not any(digit >= 5 for digit in digits):
                digits[0] = rng.choice((5, 6))
            remaining = bag_size - len(digits)
            filler_pool = tuple(set(range(10)) - GROUP_MAPPING["mid_digit"])
            digits.extend(rng.choice(filler_pool) for _ in range(remaining))
            rng.shuffle(digits)
        else:  # ternary_label == "high_digit"
            high_pool = tuple(GROUP_MAPPING["high_digit"])
            digits = [rng.choice(high_pool) for _ in range(majority_count(bag_size))]
            remaining = bag_size - len(digits)
            filler_pool = tuple(range(10))
            digits.extend(rng.choice(filler_pool) for _ in range(remaining))
            rng.shuffle(digits)

    if len(digits) != bag_size:
        raise ValueError("Digit synthesis failed to match the requested bag size.")

    derived_binary, derived_ternary = determine_labels(digits)
    if derived_binary != binary_label or derived_ternary != ternary_label:
        raise ValueError(
            "Generated digits do not satisfy requested labels: "
            f"wanted ({binary_label}, {ternary_label}) but obtained "
            f"({derived_binary}, {derived_ternary})."
        )

    return digits


def plan_label_allocation(num_slides: int, rng: random.Random) -> List[Tuple[str, str]]:
    """Return a balanced list of (binary_label, ternary_label) assignments."""

    ternary_base = num_slides // 3
    ternary_counts = {
        "low_digit": ternary_base,
        "mid_digit": ternary_base,
        "high_digit": ternary_base,
    }
    for idx, label in enumerate(("low_digit", "mid_digit", "high_digit")[: num_slides % 3]):
        ternary_counts[label] += 1

    positive_target = math.ceil(num_slides / 2)
    negative_target = num_slides - positive_target

    plans: List[Tuple[str, str]] = []

    # High-digit slides must be positive.
    for _ in range(ternary_counts["high_digit"]):
        plans.append(("positive", "high_digit"))
        positive_target -= 1

    remaining_labels = (
        ["low_digit"] * ternary_counts["low_digit"]
        + ["mid_digit"] * ternary_counts["mid_digit"]
    )

    for label in remaining_labels:
        if negative_target > 0:
            plans.append(("negative", label))
            negative_target -= 1
        else:
            plans.append(("positive", label))
            positive_target -= 1

    if positive_target != 0 or negative_target != 0:
        raise ValueError("Failed to allocate label plan with the requested balance.")

    rng.shuffle(plans)

    return plans


def write_h5(features: np.ndarray, coords: np.ndarray, destination: str) -> None:
    ensure_directory(os.path.dirname(destination))
    with h5py.File(destination, "w") as handle:
        handle.create_dataset("features", data=features.astype(np.float32))
        handle.create_dataset("coords", data=coords.astype(np.int32))


def save_shape_entry(shape_file: str, slide_id: str, width: int, height: int) -> None:
    with open(shape_file, "a", encoding="utf-8") as handle:
        handle.write(f"{slide_id},{width},{height}\n")


def stratified_kfold(
    slide_ids: Sequence[str],
    labels: Sequence[str],
    k_folds: int,
    rng: random.Random,
) -> List[List[str]]:
    """Create stratified folds while preserving label balance."""

    buckets: Dict[str, List[str]] = defaultdict(list)
    for slide_id, label in zip(slide_ids, labels):
        buckets[label].append(slide_id)

    for slides in buckets.values():
        rng.shuffle(slides)

    folds: List[List[str]] = [[] for _ in range(k_folds)]
    for slides in buckets.values():
        for index, slide_id in enumerate(slides):
            folds[index % k_folds].append(slide_id)

    for fold in folds:
        fold.sort()
    return folds


def to_wide_split(train: Sequence[str], val: Sequence[str], test: Sequence[str]) -> pd.DataFrame:
    max_len = max(len(train), len(val), len(test))
    pad = lambda seq: list(seq) + [""] * (max_len - len(seq))
    return pd.DataFrame({"train": pad(train), "val": pad(val), "test": pad(test)})


def to_boolean_split(train: Sequence[str], val: Sequence[str], test: Sequence[str]) -> pd.DataFrame:
    rows = [[slide_id, True, False, False] for slide_id in train]
    rows += [[slide_id, False, True, False] for slide_id in val]
    rows += [[slide_id, False, False, True] for slide_id in test]
    return pd.DataFrame(rows, columns=["slide_id", "train", "val", "test"])


def save_splits(
    examples: Sequence[SlideExample],
    label_accessor,
    task_name: str,
    output_dir: str,
    k_folds: int,
    rng: random.Random,
) -> None:
    """Create cross-validation CSVs compatible with ``main.py``."""

    slide_ids = [example.slide_id for example in examples]
    labels = [label_accessor(example) for example in examples]
    folds = stratified_kfold(slide_ids, labels, k_folds, rng)

    descriptor_lookup = {example.slide_id: example for example in examples}

    split_root = os.path.join(output_dir, "splits", task_name)
    ensure_directory(split_root)

    for fold_idx in range(k_folds):
        test_ids = folds[fold_idx]
        val_ids = folds[(fold_idx + 1) % k_folds]
        train_ids = [
            slide_id
            for other_idx, fold in enumerate(folds)
            if other_idx not in {fold_idx, (fold_idx + 1) % k_folds}
            for slide_id in fold
        ]
        train_ids.sort()

        wide = to_wide_split(train_ids, val_ids, test_ids)
        wide.to_csv(
            os.path.join(split_root, f"splits_{fold_idx}.csv"), index=False
        )

        boolean = to_boolean_split(train_ids, val_ids, test_ids)
        boolean.to_csv(
            os.path.join(split_root, f"splits_{fold_idx}_bool.csv"), index=False
        )

        descriptor_rows = []
        for split_name, ids in ("train", train_ids), ("val", val_ids), ("test", test_ids):
            for slide_id in ids:
                example = descriptor_lookup[slide_id]
                descriptor_rows.append(
                    {
                        "slide_id": slide_id,
                        "case_id": example.case_id,
                        "label": label_accessor(example),
                        "split": split_name,
                    }
                )

        pd.DataFrame(descriptor_rows).to_csv(
            os.path.join(split_root, f"splits_{fold_idx}_descriptor.csv"), index=False
        )


def main() -> None:
    args = parse_args()

    if args.min_patches <= 0 or args.max_patches <= 0:
        raise ValueError("Patch counts must be positive integers.")
    if args.min_patches > args.max_patches:
        raise ValueError("--min-patches cannot be larger than --max-patches.")

    ensure_directory(args.output_dir)
    h5_root = os.path.join(args.output_dir, "h5_files")
    shape_file = os.path.join(args.output_dir, "images_shape.txt")
    # Reset shape file if it already exists to avoid duplicate entries.
    if os.path.exists(shape_file):
        os.remove(shape_file)

    images, labels = load_mnist_images(args.mnist_root)
    digit_to_indices = {
        digit: np.flatnonzero(labels == digit).tolist()
        for digit in range(10)
    }

    rng = random.Random(args.seed)
    examples: List[SlideExample] = []

    label_plan = plan_label_allocation(args.num_slides, rng)

    for slide_index, (binary_label, ternary_label) in enumerate(label_plan):
        for _ in range(128):
            bag_size = rng.randint(args.min_patches, args.max_patches)
            try:
                digit_sequence = generate_digit_sequence(
                    rng, bag_size, binary_label, ternary_label
                )
            except ValueError:
                continue

            features, digit_labels = sample_slide_contents(
                rng, digit_sequence, images, digit_to_indices
            )
            coords, width, height = make_grid_coords(len(digit_labels))
            break
        else:
            raise RuntimeError(
                "Failed to synthesize a slide matching the requested label combination. "
                "Consider relaxing the patch-count bounds."
            )

        slide_id = f"slide_{slide_index:04d}"
        case_id = f"case_{slide_index // args.slides_per_case:04d}"

        write_h5(features, coords, os.path.join(h5_root, f"{slide_id}.h5"))
        save_shape_entry(shape_file, slide_id, width, height)

        examples.append(
            SlideExample(
                case_id=case_id,
                slide_id=slide_id,
                binary_label=binary_label,
                ternary_label=ternary_label,
            )
        )

    binary_df = pd.DataFrame(
        {
            "case_id": [example.case_id for example in examples],
            "slide_id": [example.slide_id for example in examples],
            "label": [example.binary_label for example in examples],
        }
    )
    binary_df.to_csv(os.path.join(args.output_dir, "mnist_binary.csv"), index=False)

    ternary_df = pd.DataFrame(
        {
            "case_id": [example.case_id for example in examples],
            "slide_id": [example.slide_id for example in examples],
            "label": [example.ternary_label for example in examples],
        }
    )
    ternary_df.to_csv(os.path.join(args.output_dir, "mnist_ternary.csv"), index=False)

    save_splits(
        examples,
        label_accessor=lambda example: example.binary_label,
        task_name="mnist_binary",
        output_dir=args.output_dir,
        k_folds=args.k_folds,
        rng=rng,
    )
    save_splits(
        examples,
        label_accessor=lambda example: example.ternary_label,
        task_name="mnist_ternary",
        output_dir=args.output_dir,
        k_folds=args.k_folds,
        rng=rng,
    )

    print(f"Synthetic dataset written to {args.output_dir}")


if __name__ == "__main__":
    main()
