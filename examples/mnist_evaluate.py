"""Launch Bayes-MIL evaluation for the synthetic MNIST dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run eval.py on MNIST checkpoints produced by main.py.',
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
    parser.add_argument('--exp-code', type=str, required=True, help='Experiment identifier used during training.')
    parser.add_argument('--results-dir', type=Path, default=REPO_ROOT / 'results')
    parser.add_argument('--seed', type=int, default=1, help='Matches the value forwarded to main.py.')
    parser.add_argument(
        '--model-type',
        choices=[
            'bmil-vis', 'bmil-addvis', 'bmil-conjvis', 'bmil-convis',
            'bmil-addenc', 'bmil-conjenc', 'bmil-conenc',
            'bmil-enc', 'bmil-spvis', 'bmil-addspvis', 'bmil-conjspvis', 'bmil-conspvis',
        ],
        default='bmil-vis',
    )
    parser.add_argument('--model-size', choices=['small', 'big'], default='small')
    parser.add_argument('--drop-out', action='store_true', help='Set if the checkpoint was trained with dropout.')
    parser.add_argument('--k', type=int, default=5, help='Number of folds to evaluate.')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')
    parser.add_argument('--fold', type=int, default=-1, help='Single fold to evaluate (default: all folds).')
    parser.add_argument('--micro-average', action='store_true', help='Use micro-average AUC for multi-class tasks.')
    parser.add_argument('--python-bin', default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    models_exp_code = f'{args.exp_code}_s{args.seed}'

    cmd = [
        args.python_bin,
        str(REPO_ROOT / 'eval.py'),
        '--data_root_dir',
        str(dataset_root),
        '--results_dir',
        str(results_dir),
        '--models_exp_code',
        models_exp_code,
        '--save_exp_code',
        models_exp_code,
        '--task',
        args.task,
        '--model_type',
        args.model_type,
        '--model_size',
        args.model_size,
        '--k',
        str(args.k),
        '--split',
        args.split,
        '--fold',
        str(args.fold),
    ]

    if args.drop_out:
        cmd.append('--drop_out')
    if args.micro_average:
        cmd.append('--micro_average')

    print(f"[MNIST evaluation] Launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == '__main__':
    main()
