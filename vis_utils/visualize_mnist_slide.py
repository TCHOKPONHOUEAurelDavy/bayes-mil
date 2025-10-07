"""Visualize synthetic MNIST slides generated for Bayes-MIL experiments.

The helper consumes the dataset layout emitted by
``processing_scripts/create_mnist_synthetic_dataset.py``. It loads a slide from
``h5_files/``, reconstructs the spatial arrangement of MNIST digits, and saves a
PNG snapshot that is convenient for quick inspections.

Example
-------

.. code-block:: bash

    python vis_utils/visualize_mnist_slide.py \\
        --dataset-root /path/to/mnist_mil_dataset \\
        --slide-id slide_0005 \\
        --output preview.png

"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PATCH_PIXELS = 28

TASK_TO_CSV: Dict[str, str] = {
    "mnist_fourbags": "mnist_fourbags.csv",
    "mnist_even_odd": "mnist_even_odd.csv",
    "mnist_adjacent_pairs": "mnist_adjacent_pairs.csv",
    "mnist_fourbags_plus": "mnist_fourbags_plus.csv",
}

TASK_DISPLAY_NAMES: Dict[str, str] = {
    "mnist_fourbags": "fourbags",
    "mnist_even_odd": "even-odd",
    "mnist_adjacent_pairs": "adjacent",
    "mnist_fourbags_plus": "fourbags+",
}


@dataclass(frozen=True)
class SlideLabels:
    """Container for the task-specific labels tied to a slide."""

    values: Dict[str, Optional[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one synthetic MNIST slide as a PNG heatmap.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory that hosts h5_files/ and metadata CSVs.",
    )
    parser.add_argument(
        "--slide-id",
        type=str,
        required=True,
        help="Identifier of the slide to visualize (e.g. slide_0001).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Destination PNG path. Defaults to <dataset-root>/visualizations/"
            "<slide-id>.png."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Dots-per-inch used when saving the figure (default: %(default)s).",
    )
    return parser.parse_args()


def default_output_path(dataset_root: str, slide_id: str) -> str:
    visual_root = os.path.join(dataset_root, "visualizations")
    os.makedirs(visual_root, exist_ok=True)
    return os.path.join(visual_root, f"{slide_id}.png")


def load_slide_arrays(dataset_root: str, slide_id: str) -> tuple[np.ndarray, np.ndarray]:
    h5_path = os.path.join(dataset_root, "h5_files", f"{slide_id}.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"Slide {slide_id!r} not found under {os.path.dirname(h5_path)!r}."
        )
    with h5py.File(h5_path, "r") as handle:
        features = handle["features"][()]
        coords = handle["coords"][()]
    if features.ndim != 2:
        raise ValueError("Expected `features` to be a 2-D array.")
    if features.shape[1] != PATCH_PIXELS * PATCH_PIXELS:
        raise ValueError(
            "This helper assumes 28x28 MNIST digits. Received feature vectors "
            f"of length {features.shape[1]}."
        )
    return features.astype(np.float32), coords.astype(np.int32)


def load_labels(dataset_root: str, slide_id: str) -> SlideLabels:
    def _read_label(csv_name: str) -> Optional[str]:
        csv_path = os.path.join(dataset_root, csv_name)
        if not os.path.exists(csv_path):
            return None
        frame = pd.read_csv(csv_path)
        matches = frame.loc[frame["slide_id"] == slide_id, "label"]
        return matches.iloc[0] if not matches.empty else None

    values: Dict[str, Optional[str]] = {}
    for task, csv_name in TASK_TO_CSV.items():
        values[task] = _read_label(csv_name)
    return SlideLabels(values=values)


def reconstruct_canvas(features: np.ndarray, coords: np.ndarray) -> np.ndarray:
    patch_size = PATCH_PIXELS
    width = int(coords[:, 0].max() + patch_size)
    height = int(coords[:, 1].max() + patch_size)
    canvas = np.zeros((height, width), dtype=np.float32)
    for patch, (x, y) in zip(features, coords):
        digit = patch.reshape(patch_size, patch_size)
        canvas[y : y + patch_size, x : x + patch_size] = digit
    return canvas


def format_title(slide_id: str, labels: SlideLabels) -> str:
    parts = [slide_id]
    for task in sorted(labels.values.keys()):
        value = labels.values[task]
        if value is not None:
            display = TASK_DISPLAY_NAMES.get(task, task)
            parts.append(f"{display}: {value}")
    return " | ".join(parts)


def save_figure(canvas: np.ndarray, output_path: str, title: str, dpi: int) -> None:
    plt.figure(figsize=(canvas.shape[1] / dpi, canvas.shape[0] / dpi), dpi=dpi)
    plt.imshow(canvas, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.dataset_root, args.slide_id)

    features, coords = load_slide_arrays(args.dataset_root, args.slide_id)
    labels = load_labels(args.dataset_root, args.slide_id)
    canvas = reconstruct_canvas(features, coords)
    title = format_title(args.slide_id, labels)
    save_figure(canvas, output_path, title, dpi=args.dpi)

    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
