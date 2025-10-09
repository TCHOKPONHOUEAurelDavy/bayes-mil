"""Generate attention heatmaps for synthetic MNIST slides.

This module can be used either as a command-line utility or imported by
automation scripts.  Compared to the previous version, the helper now supports
batch rendering so that entire dataset splits (for example the test set of a
fold) can be exported in one go, mimicking the behaviour of the original
BayesMIL heatmap pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from utils.eval_utils import initiate_model
from vis_utils.visualize_mnist_slide import (
    PATCH_PIXELS,
    SlideLabels,
    format_title,
    load_labels,
    load_slide_arrays,
    reconstruct_canvas,
)


LABEL_DICTS: Dict[str, Dict[str, int]] = {
    'mnist_fourbags': {'none': 0, 'mostly_eight': 1, 'mostly_nine': 2, 'both': 3},
    'mnist_even_odd': {'odd_majority': 0, 'even_majority': 1},
    'mnist_adjacent_pairs': {'no_adjacent_pairs': 0, 'has_adjacent_pairs': 1},
    'mnist_fourbags_plus': {'none': 0, 'three_five': 1, 'one_only': 2, 'one_and_seven': 3},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render attention heatmaps for synthetic MNIST slides.',
    )
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--task', choices=tuple(LABEL_DICTS.keys()), default='mnist_fourbags')
    parser.add_argument(
        '--model-type',
        choices=['bmil-vis', 'bmil-addvis', 'bmil-conjvis', 'bmil-convis', 'bmil-enc', 'bmil-spvis'],
        default='bmil-vis',
    )
    parser.add_argument('--model-size', choices=['small', 'big'], default='small')
    parser.add_argument('--drop-out', action='store_true', help='Set when the checkpoint was trained with dropout enabled.')
    parser.add_argument('--alpha', type=float, default=0.55, help='Opacity of the attention overlay (default: %(default)s).')
    parser.add_argument('--cmap', type=str, default='inferno', help='Matplotlib colormap for the attention map.')
    parser.add_argument('--dpi', type=int, default=200, help='Dots per inch for the saved figure.')
    parser.add_argument('--slide-id', type=str, default=None, help='Single slide to visualise (required unless --all is set).')
    parser.add_argument('--output', type=Path, default=None, help='Destination PNG path when rendering a single slide.')
    parser.add_argument('--all', action='store_true', help='Render every slide referenced in the descriptor CSV.')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test', help='Subset to export when --all is active.')
    parser.add_argument('--fold', type=int, default=0, help='Fold whose descriptor file should be used (default: %(default)s).')
    parser.add_argument(
        '--split-descriptor',
        type=Path,
        default=None,
        help='Optional path to a descriptor CSV with slide_id/split columns. Defaults to <dataset-root>/splits/<task>/splits_<fold>_descriptor.csv.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory where batch heatmaps are saved. Defaults to <dataset-root>/visualizations/<checkpoint-stem>/<split>_fold_<fold>/.',
    )
    parser.add_argument('--skip-existing', action='store_true', help='Do not overwrite heatmaps that already exist when --all is active.')
    return parser.parse_args()


def normalize(array: np.ndarray) -> np.ndarray:
    min_val = float(array.min())
    max_val = float(array.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(array)
    return (array - min_val) / (max_val - min_val)


def attention_to_canvas(attention: np.ndarray, coords: np.ndarray) -> np.ndarray:
    patch_size = PATCH_PIXELS
    width = int(coords[:, 0].max() + patch_size)
    height = int(coords[:, 1].max() + patch_size)
    canvas = np.zeros((height, width), dtype=np.float32)
    for score, (x, y) in zip(attention, coords):
        canvas[y : y + patch_size, x : x + patch_size] = score
    return canvas


def save_overlay(
    base_canvas: np.ndarray,
    attention_map: np.ndarray,
    output_path: Path,
    title: str,
    alpha: float,
    cmap: str,
    dpi: int,
) -> None:
    plt.figure(figsize=(base_canvas.shape[1] / dpi, base_canvas.shape[0] / dpi), dpi=dpi)
    plt.imshow(base_canvas, cmap='gray', interpolation='nearest')
    plt.imshow(attention_map, cmap=cmap, alpha=alpha, interpolation='nearest')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def default_single_output(dataset_root: Path, slide_id: str) -> Path:
    output_dir = dataset_root / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f'{slide_id}_attention.png'


def default_batch_output(dataset_root: Path, checkpoint: Path, split: str, fold: int) -> Path:
    split_suffix = f'{split}_fold_{fold}' if split != 'all' else f'all_fold_{fold}'
    output_dir = dataset_root / 'visualizations' / checkpoint.stem / split_suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_descriptor(dataset_root: Path, task: str, fold: int, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    descriptor = dataset_root / 'splits' / task / f'splits_{fold}_descriptor.csv'
    if not descriptor.exists():
        raise FileNotFoundError(
            f'Unable to locate descriptor CSV at {descriptor}. Use --split-descriptor to provide a custom file.'
        )
    return descriptor


def load_split_slides(descriptor: Path, split: str) -> List[str]:
    frame = pd.read_csv(descriptor)
    if 'slide_id' not in frame.columns or 'split' not in frame.columns:
        raise ValueError(
            f'Descriptor {descriptor} must contain slide_id and split columns.'
        )
    if split != 'all':
        frame = frame.loc[frame['split'] == split]
    slide_ids = frame['slide_id'].astype(str).tolist()
    if not slide_ids:
        raise RuntimeError(
            f'No slides found for split={split!r} in {descriptor}. Ensure the dataset was generated with matching folds.'
        )
    return slide_ids


def run_inference(
    model: torch.nn.Module,
    model_type: str,
    features: np.ndarray,
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    features_tensor = torch.from_numpy(features).to(device)

    with torch.no_grad():
        if model_type == 'bmil-spvis':
            width = int(coords[:, 0].max() + PATCH_PIXELS)
            height = int(coords[:, 1].max() + PATCH_PIXELS)
            _, _, _, slide_prob, attention = model(
                features_tensor,
                coords,
                height,
                width,
                validation=True,
            )
            bag_probs = slide_prob
        else:
            _, _, _, bag_probs, attention = model(features_tensor, validation=True)

    probs = bag_probs.squeeze(0).detach().cpu().numpy()
    attention_scores = attention.squeeze(0).detach().cpu().numpy()
    return probs, attention_scores


@dataclass
class HeatmapResult:
    slide_id: str
    output_path: Path
    predicted_label: str
    predicted_probability: float
    probabilities: Sequence[float]
    labels: SlideLabels


class MNISTHeatmapRenderer:
    """Utility that holds a loaded model and renders heatmaps on demand."""

    def __init__(
        self,
        dataset_root: Path,
        checkpoint: Path,
        *,
        task: str,
        model_type: str,
        model_size: str = 'small',
        drop_out: bool = False,
    ) -> None:
        self.dataset_root = dataset_root.expanduser().resolve()
        self.checkpoint = checkpoint.expanduser().resolve()
        if not self.checkpoint.exists():
            raise FileNotFoundError(f'Checkpoint not found at {self.checkpoint}')
        if task not in LABEL_DICTS:
            raise ValueError(f'Unsupported MNIST task {task!r}')

        self.task = task
        self.model_type = model_type
        self.label_dict = LABEL_DICTS[task]
        self.inv_label_dict = {idx: name for name, idx in self.label_dict.items()}

        checkpoint_state = torch.load(self.checkpoint, map_location='cpu')

        def resolve_state_dict(raw_state):
            if isinstance(raw_state, dict):
                for key in ('state_dict', 'model_state_dict', 'model'):
                    nested = raw_state.get(key)
                    if isinstance(nested, dict):
                        return nested
                return raw_state
            raise TypeError(
                f'Unexpected checkpoint format in {self.checkpoint}: expected a mapping, got {type(raw_state)!r}'
            )

        state_dict = resolve_state_dict(checkpoint_state)

        classifier_weight_key = None
        for key in state_dict.keys():
            if key.endswith('classifiers.weight'):
                classifier_weight_key = key
                break

        if classifier_weight_key is None:
            available = ', '.join(sorted(state_dict.keys()))
            raise KeyError(
                'Could not locate classifiers.weight in checkpoint '
                f'{self.checkpoint}. '
                f'Available keys: {available}'
            )

        checkpoint_classes = int(state_dict[classifier_weight_key].shape[0])
        expected_classes = len(self.label_dict)
        if checkpoint_classes != expected_classes:
            raise ValueError(
                f'Checkpoint {self.checkpoint} contains a classifier trained with '
                f'{checkpoint_classes} classes, but task {task!r} expects '
                f'{expected_classes}. Ensure you are passing the correct --task '
                'and checkpoint combination.'
            )

        init_args = SimpleNamespace(
            model_type=model_type,
            drop_out=drop_out,
            n_classes=checkpoint_classes,
            model_size=model_size,
        )

        self.model, _ = initiate_model(
            init_args,
            str(self.checkpoint),
            feature_dim=PATCH_PIXELS * PATCH_PIXELS,
        )
        self.model.eval()

    def render_slide(
        self,
        slide_id: str,
        *,
        alpha: float,
        cmap: str,
        dpi: int,
        output_path: Path | None = None,
    ) -> HeatmapResult:
        features, coords = load_slide_arrays(str(self.dataset_root), slide_id)
        labels = load_labels(str(self.dataset_root), slide_id)
        base_canvas = reconstruct_canvas(features, coords)

        probs, attention = run_inference(self.model, self.model_type, features, coords)
        pred_idx = int(np.argmax(probs))
        pred_label = self.inv_label_dict[pred_idx]
        pred_prob = float(probs[pred_idx])

        attention_canvas = normalize(attention_to_canvas(attention, coords))
        destination = output_path or default_single_output(self.dataset_root, slide_id)
        destination = destination.expanduser().resolve()

        title = f"{format_title(slide_id, labels)} | predicted: {pred_label} ({pred_prob:.2f})"
        save_overlay(base_canvas, attention_canvas, destination, title, alpha, cmap, dpi)

        return HeatmapResult(
            slide_id=slide_id,
            output_path=destination,
            predicted_label=pred_label,
            predicted_probability=pred_prob,
            probabilities=probs.tolist(),
            labels=labels,
        )

    def batch_render(
        self,
        slide_ids: Iterable[str],
        *,
        alpha: float,
        cmap: str,
        dpi: int,
        output_dir: Path,
        skip_existing: bool = False,
    ) -> List[HeatmapResult]:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        results: List[HeatmapResult] = []
        for slide_id in slide_ids:
            destination = output_dir / f'{slide_id}_attention.png'
            if skip_existing and destination.exists():
                print(f'[MNIST heatmaps] Skipping existing file {destination}')
                continue
            result = self.render_slide(
                slide_id,
                alpha=alpha,
                cmap=cmap,
                dpi=dpi,
                output_path=destination,
            )
            results.append(result)
        return results


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()

    renderer = MNISTHeatmapRenderer(
        dataset_root,
        checkpoint_path,
        task=args.task,
        model_type=args.model_type,
        model_size=args.model_size,
        drop_out=args.drop_out,
    )

    if args.all:
        descriptor = resolve_descriptor(dataset_root, args.task, args.fold, args.split_descriptor)
        slide_ids = load_split_slides(descriptor, args.split)
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else default_batch_output(dataset_root, checkpoint_path, args.split, args.fold)
        )
        results = renderer.batch_render(
            slide_ids,
            alpha=args.alpha,
            cmap=args.cmap,
            dpi=args.dpi,
            output_dir=output_dir,
            skip_existing=args.skip_existing,
        )
        print(
            f'Saved {len(results)} heatmaps to {output_dir}. '
            f'Descriptor: {descriptor} | split={args.split}'
        )
    else:
        if args.slide_id is None:
            raise ValueError('Specify --slide-id when not running with --all.')
        output_path = (
            args.output.expanduser().resolve()
            if args.output is not None
            else default_single_output(dataset_root, args.slide_id)
        )
        result = renderer.render_slide(
            args.slide_id,
            alpha=args.alpha,
            cmap=args.cmap,
            dpi=args.dpi,
            output_path=output_path,
        )
        prob_strings = ', '.join(
            f"{renderer.inv_label_dict[i]}: {result.probabilities[i]:.3f}"
            for i in range(len(result.probabilities))
        )
        print(f'Saved attention heatmap to {result.output_path}')
        print(f'Class probabilities -> {prob_strings}')


if __name__ == '__main__':
    main()
