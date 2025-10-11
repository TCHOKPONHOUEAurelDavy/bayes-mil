"""Inspect the synthetic MNIST MIL dataset from a small Python script.

The helper demonstrates how to:

* load one of the generated MNIST CSV descriptors,
* filter a fold/split combination using the boolean split files, and
* iterate over the resulting slides with ``iter_explainability_batches``.

This mirrors the behaviour expected by the explainability utilities so it can
be used as a starting point for custom experiments or notebooks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from datasets.dataset_generic import Generic_MIL_Dataset, Generic_Split
from utils.explainability_utils import iter_explainability_batches


def _build_label_dict(csv_path: Path) -> dict[str, int]:
    """Recover the mapping from human-readable labels to class indices."""

    frame = pd.read_csv(csv_path)
    unique = frame.drop_duplicates("label_name")
    mapping = {str(row.label_name): int(row.label) for row in unique.itertuples()}
    if not mapping:
        raise ValueError(f"No labels found in {csv_path}")
    return mapping


def _load_split(
    dataset_root: Path,
    task: str,
    split: str,
    fold: int,
) -> Generic_Split:
    """Load one MNIST split as a ``Generic_Split`` dataset instance."""

    csv_path = dataset_root / f"{task}.csv"
    label_dict = _build_label_dict(csv_path)

    base_dataset = Generic_MIL_Dataset(
        csv_path=str(csv_path),
        data_dir=str(dataset_root),
        shuffle=False,
        print_info=False,
        label_dict=label_dict,
        patient_strat=False,
        ignore=[],
        label_col="label_name",
    )

    split_csv = dataset_root / "splits" / task / f"splits_{fold}_bool.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Could not find split descriptor {split_csv}")
    split_frame = pd.read_csv(split_csv)
    if split not in split_frame.columns:
        raise KeyError(f"Split column {split!r} not present in {split_csv}")

    mask = split_frame[split].fillna(False).astype(bool)
    slide_ids = set(split_frame.loc[mask, "slide_id"].dropna().astype(str))

    filtered = base_dataset.slide_data[
        base_dataset.slide_data["slide_id"].isin(slide_ids)
    ].reset_index(drop=True)

    return Generic_Split(
        filtered,
        data_dir=str(dataset_root),
        shape_dict=base_dataset.shape_dict,
        num_classes=base_dataset.num_classes,
        use_h5=True,
    )


def _preview_batches(dataset: Generic_Split, limit: int | None) -> Iterable[str]:
    """Yield human-readable descriptions of the first ``limit`` slides."""

    for index, batch in enumerate(iter_explainability_batches(dataset), start=1):
        feature_shape = tuple(batch["features"].shape)
        numbers_info = (
            f"numbers shape={batch['numbers'].shape}" if batch.get("numbers") is not None else "numbers=missing"
        )
        evidence_info = (
            f"evidence keys={sorted(batch['evidence'].keys())}" if batch.get("evidence") else "evidence=missing"
        )
        yield (
            f"{index:02d}. slide_id={batch['slide_id']} label={batch['label']} "
            f"features={feature_shape} {numbers_info} {evidence_info}"
        )
        if limit is not None and index >= limit:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview slides from the synthetic MNIST MIL dataset.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory produced by processing_scripts/create_mnist_synthetic_dataset.py",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "mnist_fourbags",
            "mnist_even_odd",
            "mnist_adjacent_pairs",
            "mnist_fourbags_plus",
        ],
        default="mnist_fourbags",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Which split column to sample from the boolean split file.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Cross-validation fold to inspect (defaults to the first fold).",
    )
    parser.add_argument(
        "--max-slides",
        type=int,
        default=5,
        help="Limit the number of slides printed to the console (use -1 for all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    dataset = _load_split(dataset_root, args.task, args.split, args.fold)

    limit = None if args.max_slides == -1 else max(args.max_slides, 0)

    print(
        f"Loaded {len(dataset)} slides from {args.task} "
        f"(fold={args.fold}, split={args.split})."
    )
    for line in _preview_batches(dataset, limit):
        print(line)


if __name__ == "__main__":
    main()
