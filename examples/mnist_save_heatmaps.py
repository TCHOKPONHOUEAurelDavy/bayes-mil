"""Export attention heatmaps for a full MNIST split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure the repository root (which contains the ``vis_utils`` package) is on the path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vis_utils.mnist_attention_heatmap import (  # noqa: E402
    MNISTHeatmapRenderer,
    load_split_slides,
    resolve_descriptor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Save attention heatmaps for every slide in a MNIST split.',
    )
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument(
        '--task',
        choices=(
            'mnist_fourbags',
            'mnist_even_odd',
            'mnist_adjacent_pairs',
            'mnist_fourbags_plus',
        ),
        default='mnist_fourbags',
    )
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--exp-code', type=str, required=True, help='Experiment identifier used during training.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument(
        '--model-type',
        choices=['bmil-vis', 'bmil-addvis', 'bmil-conjvis', 'bmil-convis', 'bmil-enc', 'bmil-spvis'],
        default='bmil-vis',
    )
    parser.add_argument('--model-size', choices=['small', 'big'], default='small')
    parser.add_argument('--drop-out', action='store_true')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')
    parser.add_argument('--split-descriptor', type=Path, default=None, help='Override path to splits_<fold>_descriptor.csv.')
    parser.add_argument('--output-dir', type=Path, default=None, help='Directory where PNGs will be written.')
    parser.add_argument('--alpha', type=float, default=0.55)
    parser.add_argument('--cmap', type=str, default='inferno')
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--skip-existing', action='store_true')
    return parser.parse_args()


def default_output_dir(dataset_root: Path, exp_code: str, seed: int, split: str, fold: int) -> Path:
    split_suffix = f'{split}_fold_{fold}' if split != 'all' else f'all_fold_{fold}'
    return dataset_root / 'visualizations' / f'{exp_code}_s{seed}' / split_suffix


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    exp_name = f'{args.exp_code}_s{args.seed}'
    checkpoint_path = results_dir / exp_name / f's_{args.fold}_checkpoint.pt'

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f'Checkpoint not found at {checkpoint_path}. Ensure training finished for fold {args.fold}.'
        )

    renderer = MNISTHeatmapRenderer(
        dataset_root,
        checkpoint_path,
        task=args.task,
        model_type=args.model_type,
        model_size=args.model_size,
        drop_out=args.drop_out,
    )

    descriptor = resolve_descriptor(dataset_root, args.task, args.fold, args.split_descriptor)
    slide_ids = load_split_slides(descriptor, args.split)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(dataset_root, args.exp_code, args.seed, args.split, args.fold)
    )

    results = renderer.batch_render(
        slide_ids,
        alpha=args.alpha,
        cmap=args.cmap,
        dpi=args.dpi,
        output_dir=output_dir,
        skip_existing=args.skip_existing,
    )

    summary_rows = []
    for result in results:
        summary_rows.append(
            {
                'slide_id': result.slide_id,
                'predicted_label': result.predicted_label,
                'predicted_probability': result.predicted_probability,
                'output_path': str(result.output_path),
            }
        )

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = output_dir / 'heatmap_summary.csv'
        summary.to_csv(summary_path, index=False)
        print(f'Saved {len(results)} heatmaps and wrote metadata to {summary_path}')
    else:
        print('No new heatmaps generated (skip-existing was enabled and files were present).')


if __name__ == '__main__':
    main()
