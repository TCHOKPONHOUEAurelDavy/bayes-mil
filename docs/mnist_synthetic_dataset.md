# Synthetic MNIST MIL dataset

This guide explains how to generate a toy dataset based on MNIST digits that
follows the file layout expected by Bayes-MIL. The dataset is convenient for
running fast experiments without relying on whole-slide images.

## 1. Create the dataset

Run the helper script and provide the output directory that will host the
artifacts. The script downloads MNIST through `torchvision` on first use.

```bash
python processing_scripts/create_mnist_synthetic_dataset.py \
    --output-dir /path/to/mnist_mil_dataset \
    --num-slides 200 --k-folds 5
```

Key options:

- `--num-slides`: how many synthetic “slides” (bags) to create.
- `--min-patches` / `--max-patches`: range for the number of MNIST digits per slide.
- `--slides-per-case`: how many slides share the same case identifier.
- `--k-folds`: number of cross-validation folds saved under `splits/<task>/`.

The directory will contain:

- `h5_files/slide_xxxx.h5`: flattened MNIST pixels and 2-D coordinates for each bag.
- `mnist_binary.csv`: metadata for the binary task (`negative` vs `positive`).
- `mnist_ternary.csv`: metadata for the ternary task (`low_digit`, `mid_digit`, `high_digit`).
- `images_shape.txt`: synthetic canvas sizes used when reconstructing spatial maps.
- `splits/mnist_binary/` and `splits/mnist_ternary/`: cross-validation CSV files.
- The generator enforces that each binary and ternary label is represented (when the
  requested number of slides allows for it), avoiding degenerate datasets lacking a class.

## 2. Run the Bayes-MIL pipeline step by step

The following helpers mirror the manual commands usually issued against
`main.py`, `eval.py`, and the visualisation utilities. They are provided to keep
the workflow explicit and modular.

### 2.1 Train

```bash
python examples/mnist_train.py \
    --dataset-root /path/to/mnist_mil_dataset \
    --task mnist_binary \
    --exp-code mnist_demo \
    --k 5 --max-epochs 20 --lr 5e-4
```

All flags map one-to-one to the arguments consumed by `main.py`, so you can add
`--drop-out`, `--early-stopping`, or `--weighted-sample` as needed. The script
writes checkpoints under `<results-dir>/<exp-code>_s<seed>/`.

### 2.2 Evaluate

After training, evaluate the checkpoints with `eval.py` via:

```bash
python examples/mnist_evaluate.py \
    --dataset-root /path/to/mnist_mil_dataset \
    --results-dir results \
    --task mnist_binary \
    --exp-code mnist_demo \
    --k 5
```

Use `--split` to inspect `train`, `val`, `test`, or `all`, and `--fold` to
restrict evaluation to a single fold. The helper automatically points to the
MNIST split directory so no additional configuration is required.

### 2.3 Save heatmaps

To mimic the original BayesMIL behaviour and export heatmaps for every slide in
the test set of one fold, run:

```bash
python examples/mnist_save_heatmaps.py \
    --dataset-root /path/to/mnist_mil_dataset \
    --results-dir results \
    --task mnist_binary \
    --exp-code mnist_demo \
    --fold 0 --split test
```

The script writes PNGs to
`<dataset-root>/visualizations/<exp-code>_s<seed>/test_fold_<fold>/` and emits a
`heatmap_summary.csv` with predicted labels. Use `--split val` or `--split train`
to export other subsets, or `--split all` to process every slide contained in
the descriptor CSV. Pass `--skip-existing` to avoid re-rendering PNGs that are
already present.

## 3. Troubleshooting

- Ensure that `torchvision` is installed in the active environment so the script
  can download MNIST.
- When re-generating the dataset in the same directory, old files are overwritten.
- If you wish to experiment with custom label definitions, modify the helper
  script before generating the dataset.

### Heatmap visualisation only

If you already have a trained checkpoint, the dedicated utility below loads the
model and saves an attention overlay for one slide:

```bash
python vis_utils/mnist_attention_heatmap.py \
    --dataset-root /path/to/mnist_mil_dataset \
    --checkpoint results/mnist_demo_s1/s_0_checkpoint.pt \
    --task mnist_binary \
    --model-type bmil-vis \
    --slide-id slide_0000
```

The output PNG is written to `<dataset-root>/visualizations/<slide-id>_attention.png`
unless `--output` specifies a different destination. Use the new `--all`
switch to render an entire split directly from the utility:

```bash
python vis_utils/mnist_attention_heatmap.py \
    --dataset-root /path/to/mnist_mil_dataset \
    --checkpoint results/mnist_demo_s1/s_0_checkpoint.pt \
    --task mnist_binary \
    --model-type bmil-vis \
    --fold 0 --split test --all
```
