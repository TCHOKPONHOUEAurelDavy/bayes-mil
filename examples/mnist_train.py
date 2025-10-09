"""Convenience launcher for training Bayes-MIL on the synthetic MNIST dataset.

The helper mirrors the command structure documented in ``docs/mnist_synthetic_dataset.md``
and simply forwards the collected arguments to ``main.py``.  Keeping the logic in
its own script allows the user to trigger only the training stage when following
the pipeline step by step.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Launch Bayes-MIL training on the synthetic MNIST dataset.',
    )
    parser.add_argument('--dataset-root', type=Path, required=True, help='Directory produced by create_mnist_synthetic_dataset.py.')
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
    parser.add_argument('--exp-code', type=str, default='mnist_demo', help='Experiment identifier forwarded to main.py.')
    parser.add_argument('--results-dir', type=Path, default=REPO_ROOT / 'results', help='Where checkpoints will be written.')
    parser.add_argument(
        '--model-type',
        choices=['bmil-vis', 'bmil-addvis', 'bmil-conjvis', 'bmil-convis', 'bmil-enc', 'bmil-spvis'],
        default='bmil-vis',
    )
    parser.add_argument('--model-size', choices=['small', 'big'], default='small')
    parser.add_argument('--k', type=int, default=5, help='Number of cross-validation folds.')
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--drop-out', action='store_true', help='Enable dropout in the MIL model.')
    parser.add_argument('--early-stopping', action='store_true', help='Forward --early_stopping to main.py.')
    parser.add_argument('--weighted-sample', action='store_true', help='Enable --weighted_sample when training.')
    parser.add_argument('--python-bin', default=sys.executable, help='Python interpreter used to execute main.py.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()

    cmd = [
        args.python_bin,
        str(REPO_ROOT / 'main.py'),
        '--data_root_dir',
        str(dataset_root),
        '--task',
        args.task,
        '--model_type',
        args.model_type,
        '--model_size',
        args.model_size,
        '--exp_code',
        args.exp_code,
        '--results_dir',
        str(results_dir),
        '--k',
        str(args.k),
        '--max_epochs',
        str(args.max_epochs),
        '--lr',
        str(args.lr),
        '--seed',
        str(args.seed),
    ]

    if args.drop_out:
        cmd.append('--drop_out')
    if args.early_stopping:
        cmd.append('--early_stopping')
    if args.weighted_sample:
        cmd.append('--weighted_sample')

    print(f"[MNIST training] Launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == '__main__':
    main()
