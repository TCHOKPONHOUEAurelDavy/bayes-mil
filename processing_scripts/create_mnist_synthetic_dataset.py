"""Utility to generate synthetic MIL datasets from MNIST.

The resulting directory mimics the structure expected by ``Generic_MIL_Dataset``:

* ``h5_files`` contains one HDF5 file per synthetic "slide".
* ``mnist_fourbags.csv`` stores slide-level labels for the Four-Bags task.
* ``mnist_even_odd.csv`` stores slide-level labels for the even/odd majority task.
* ``mnist_adjacent_pairs.csv`` stores slide-level labels for the adjacent-pairs task.
* ``mnist_fourbags_plus.csv`` stores slide-level labels for the Four-Bags-Plus task.
* ``splits/<task>/`` holds cross-validation splits aligned with ``main.py``.
* ``images_shape.txt`` records the spatial canvas size for every slide.

Example
-------

.. code-block:: bash

    python processing_scripts/create_mnist_synthetic_dataset.py \
        --output-dir data/mnist_mil --num-slides 150 --k-folds 5

The command above creates a dataset that can be consumed by the training
entry-point with the new tasks (``mnist_fourbags``, ``mnist_even_odd``,
``mnist_adjacent_pairs``, and ``mnist_fourbags_plus``).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

# Allow running the script directly via an absolute path by ensuring the
# repository root (which contains the ``processing_scripts`` package) is on the
# Python module search path.
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from processing_scripts.mnist_interpretability_tasks import (  # noqa: E402
    EvidenceBundle,
    TASK_METADATA_FNS,
)


TASK_LABEL_MAPS = {
    "mnist_fourbags": {
        0: "none",
        1: "mostly_eight",
        2: "mostly_nine",
        3: "both",
    },
    "mnist_even_odd": {
        0: "odd_majority",
        1: "even_majority",
    },
    "mnist_adjacent_pairs": {
        0: "no_adjacent_pairs",
        1: "has_adjacent_pairs",
    },
    "mnist_fourbags_plus": {
        0: "none",
        1: "three_five",
        2: "one_only",
        3: "one_and_seven",
    },
}

@dataclass(frozen=True)
class TaskSlideExample:
    """Container linking a slide to the label of a specific task."""

    case_id: str
    slide_id: str
    label: str


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
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=sorted(TASK_LABEL_MAPS.keys()),
        help=(
            "Subset of MNIST interpretability tasks to generate. "
            "Defaults to all tasks when omitted."
        ),
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


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def bundle_to_dict(
    bundle: EvidenceBundle, numbers: Sequence[int], label: str
) -> Dict[str, Any]:
    """Convert an :class:`EvidenceBundle` to a serialisable dictionary."""

    evidence = {
        cls: tensor.detach().cpu().numpy().tolist()
        for cls, tensor in bundle.evidence.items()
    }
    if bundle.instance_labels is not None:
        instance_labels = bundle.instance_labels.detach().cpu().numpy().tolist()
    else:
        instance_labels = None
    return {
        "numbers": list(numbers),
        "target": bundle.target,
        "label": label,
        "evidence": evidence,
        "instance_labels": instance_labels,
    }


def write_h5(features: np.ndarray, coords: np.ndarray, destination: str) -> None:
    ensure_directory(os.path.dirname(destination))
    with h5py.File(destination, "w") as handle:
        handle.create_dataset("features", data=features.astype(np.float32))
        handle.create_dataset("coords", data=coords.astype(np.int32))


def save_shape_entry(shape_file: str, slide_id: str, width: int, height: int) -> None:
    with open(shape_file, "a", encoding="utf-8") as handle:
        handle.write(f"{slide_id},{width},{height}\n")


def compute_class_counts(
    slides: Sequence[Mapping[str, Any]], selected_tasks: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {
        task: {label: 0 for label in TASK_LABEL_MAPS[task].values()}
        for task in selected_tasks
    }
    for slide in slides:
        numbers_tensor = torch.tensor(slide["numbers"], dtype=torch.long)
        for task in selected_tasks:
            bundle = TASK_METADATA_FNS[task](numbers_tensor)
            label_map = TASK_LABEL_MAPS[task]
            label = label_map.get(bundle.target)
            if label is None:
                raise KeyError(
                    f"Task {task} does not provide a label mapping for target {bundle.target}."
                )
            counts[task][label] += 1
    return counts


def find_missing_labels(
    class_counts: Mapping[str, Mapping[str, int]]
) -> Dict[str, List[str]]:
    missing: Dict[str, List[str]] = {}
    for task_name, labels in class_counts.items():
        absent = [label for label, count in labels.items() if count == 0]
        if absent:
            missing[task_name] = absent
    return missing


def _choose_bag_size(
    rng: random.Random, min_patches: int, max_patches: int, min_required: int
) -> int:
    lower = max(min_patches, min_required)
    upper = max(max_patches, lower)
    return rng.randint(lower, upper)


def targeted_digit_sequence(
    task_name: str,
    label: str,
    min_patches: int,
    max_patches: int,
    rng: random.Random,
) -> List[int]:
    if task_name == "mnist_fourbags":
        if label == "none":
            base = [rng.choice([d for d in range(10) if d not in {8, 9}])]
            filler = [d for d in range(10) if d not in {8, 9}]
        elif label == "mostly_eight":
            base = [8]
            filler = [d for d in range(10) if d != 9]
        elif label == "mostly_nine":
            base = [9]
            filler = [d for d in range(10) if d != 8]
        elif label == "both":
            base = [8, 9]
            filler = list(range(10))
        else:
            raise ValueError(f"Unknown label {label} for task {task_name}.")
    elif task_name == "mnist_even_odd":
        if label == "odd_majority":
            base = [1]
            filler = [1, 3, 5, 7, 9]
        elif label == "even_majority":
            base = [2]
            filler = [0, 2, 4, 6, 8]
        else:
            raise ValueError(f"Unknown label {label} for task {task_name}.")
    elif task_name == "mnist_adjacent_pairs":
        if label == "has_adjacent_pairs":
            base = [0, 1]
            filler = [0, 1, 5, 6, 7, 8, 9]
        elif label == "no_adjacent_pairs":
            base = [0]
            filler = [0, 5, 6, 7, 8, 9]
        else:
            raise ValueError(f"Unknown label {label} for task {task_name}.")
    elif task_name == "mnist_fourbags_plus":
        if label == "none":
            base = [0]
            filler = [0, 2, 4, 6, 8, 9]
        elif label == "three_five":
            base = [3, 5]
            filler = [0, 2, 4, 6, 8, 9, 3, 5]
        elif label == "one_only":
            base = [1]
            filler = [0, 1, 2, 4, 6, 8, 9]
        elif label == "one_and_seven":
            base = [1, 7]
            filler = [0, 1, 2, 4, 6, 7, 8, 9]
        else:
            raise ValueError(f"Unknown label {label} for task {task_name}.")
    else:
        raise ValueError(f"Unknown task {task_name}.")

    bag_size = _choose_bag_size(rng, min_patches, max_patches, len(base))
    digits = list(base)
    while len(digits) < bag_size:
        digits.append(rng.choice(filler))
    rng.shuffle(digits)
    return digits


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
    examples: Sequence[TaskSlideExample],
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


def generate_task_dataset(
    slides: Sequence[Dict[str, Any]],
    output_dir: str,
    k_folds: int,
    seed: int,
    task_name: str,
    metadata_fn: Callable[[torch.Tensor], EvidenceBundle],
) -> List[TaskSlideExample]:
    """Generate dataset artefacts for a single MNIST interpretability task."""

    label_map = TASK_LABEL_MAPS[task_name]
    evidence_dir = os.path.join(output_dir, "evidence", task_name)
    ensure_directory(evidence_dir)

    examples: List[TaskSlideExample] = []
    for slide in slides:
        numbers_tensor = torch.tensor(slide["numbers"], dtype=torch.long)
        bundle = metadata_fn(numbers_tensor)
        label = label_map.get(bundle.target)
        if label is None:
            raise KeyError(
                f"Task {task_name} does not provide a label mapping for target {bundle.target}."
            )

        payload = bundle_to_dict(bundle, slide["numbers"], label)
        torch.save(payload, os.path.join(evidence_dir, f"{slide['slide_id']}.pt"))

        examples.append(
            TaskSlideExample(
                case_id=slide["case_id"],
                slide_id=slide["slide_id"],
                label=label,
            )
        )

    df = pd.DataFrame(
        {
            "case_id": [example.case_id for example in examples],
            "slide_id": [example.slide_id for example in examples],
            "label": [example.label for example in examples],
        }
    )
    df.to_csv(os.path.join(output_dir, f"{task_name}.csv"), index=False)

    save_splits(
        examples,
        label_accessor=lambda example: example.label,
        task_name=task_name,
        output_dir=output_dir,
        k_folds=k_folds,
        rng=random.Random(seed),
    )

    return examples


def generate_mnist_fourbags_dataset(
    slides: Sequence[Dict[str, Any]], output_dir: str, k_folds: int, seed: int
) -> List[TaskSlideExample]:
    return generate_task_dataset(
        slides,
        output_dir,
        k_folds,
        seed,
        "mnist_fourbags",
        TASK_METADATA_FNS["mnist_fourbags"],
    )


def generate_mnist_even_odd_dataset(
    slides: Sequence[Dict[str, Any]], output_dir: str, k_folds: int, seed: int
) -> List[TaskSlideExample]:
    return generate_task_dataset(
        slides,
        output_dir,
        k_folds,
        seed,
        "mnist_even_odd",
        TASK_METADATA_FNS["mnist_even_odd"],
    )


def generate_mnist_adjacent_pairs_dataset(
    slides: Sequence[Dict[str, Any]], output_dir: str, k_folds: int, seed: int
) -> List[TaskSlideExample]:
    return generate_task_dataset(
        slides,
        output_dir,
        k_folds,
        seed,
        "mnist_adjacent_pairs",
        TASK_METADATA_FNS["mnist_adjacent_pairs"],
    )


def generate_mnist_fourbags_plus_dataset(
    slides: Sequence[Dict[str, Any]], output_dir: str, k_folds: int, seed: int
) -> List[TaskSlideExample]:
    return generate_task_dataset(
        slides,
        output_dir,
        k_folds,
        seed,
        "mnist_fourbags_plus",
        TASK_METADATA_FNS["mnist_fourbags_plus"],
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

    evidence_root = os.path.join(args.output_dir, "evidence")
    if os.path.exists(evidence_root):
        shutil.rmtree(evidence_root)

    if args.tasks is None:
        selected_tasks = list(TASK_LABEL_MAPS.keys())
    else:
        # ``choices`` already ensures validity; keep ordering deterministic.
        selected_tasks = list(dict.fromkeys(args.tasks))

    rng = random.Random(args.seed)
    slides: List[Dict[str, Any]] = []
    for _ in range(args.num_slides):
        bag_size = rng.randint(args.min_patches, args.max_patches)
        digit_sequence = [rng.randrange(10) for _ in range(bag_size)]
        features, digit_labels = sample_slide_contents(
            rng, digit_sequence, images, digit_to_indices
        )
        coords, width, height = make_grid_coords(len(digit_labels))
        slides.append(
            {
                "features": features,
                "coords": coords,
                "width": width,
                "height": height,
                "numbers": digit_labels,
            }
        )

    class_counts = compute_class_counts(slides, selected_tasks)
    missing = find_missing_labels(class_counts)
    extra_slides = 0
    while missing:
        for task_name, missing_labels in missing.items():
            for label in missing_labels:
                digit_sequence = targeted_digit_sequence(
                    task_name,
                    label,
                    args.min_patches,
                    args.max_patches,
                    rng,
                )
                features, digit_labels = sample_slide_contents(
                    rng, digit_sequence, images, digit_to_indices
                )
                coords, width, height = make_grid_coords(len(digit_labels))
                numbers_tensor = torch.tensor(digit_labels, dtype=torch.long)
                bundle = TASK_METADATA_FNS[task_name](numbers_tensor)
                observed_label = TASK_LABEL_MAPS[task_name][bundle.target]
                if observed_label != label:
                    raise RuntimeError(
                        "Targeted synthesis produced label "
                        f"{observed_label} for task {task_name} instead of expected {label}."
                    )
                slides.append(
                    {
                        "features": features,
                        "coords": coords,
                        "width": width,
                        "height": height,
                        "numbers": digit_labels,
                    }
                )
                extra_slides += 1

        class_counts = compute_class_counts(slides, selected_tasks)
        missing = find_missing_labels(class_counts)

    if extra_slides:
        print(
            "Added"
            f" {extra_slides} extra slide{'s' if extra_slides != 1 else ''}"
            " to ensure every selected task observed all classes."
        )

    slides_with_ids: List[Dict[str, Any]] = []
    for slide_index, slide in enumerate(slides):
        slide_id = f"slide_{slide_index:04d}"
        case_id = f"case_{slide_index // args.slides_per_case:04d}"

        write_h5(slide["features"], slide["coords"], os.path.join(h5_root, f"{slide_id}.h5"))
        save_shape_entry(shape_file, slide_id, slide["width"], slide["height"])

        slides_with_ids.append({**slide, "slide_id": slide_id, "case_id": case_id})

    task_generators: Dict[
        str,
        Callable[[Sequence[Dict[str, Any]], str, int, int], List[TaskSlideExample]],
    ] = {
        "mnist_fourbags": generate_mnist_fourbags_dataset,
        "mnist_even_odd": generate_mnist_even_odd_dataset,
        "mnist_adjacent_pairs": generate_mnist_adjacent_pairs_dataset,
        "mnist_fourbags_plus": generate_mnist_fourbags_plus_dataset,
    }

    for task_name in selected_tasks:
        task_generators[task_name](
            slides_with_ids,
            args.output_dir,
            args.k_folds,
            args.seed,
        )

    print(f"Synthetic dataset written to {args.output_dir}")


if __name__ == "__main__":
    main()
