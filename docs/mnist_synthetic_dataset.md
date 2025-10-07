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
    --task mnist_fourbags --num-slides 200 --k-folds 5
```

Key options:

- `--task`: which interpretability dataset to generate. Run the script again with a
  different task name to create the other variants independently.
- `--num-slides`: target number of synthetic “slides” (bags) to create. The script
  may append a few extra slides so that the minority class still represents at
  least 25% of the final dataset.
- `--min-patches` / `--max-patches`: range for the number of MNIST digits per slide.
- `--slides-per-case`: how many slides share the same case identifier.
- `--k-folds`: number of cross-validation folds saved under `splits/<task>/`.

The directory will contain:

- `h5_files/slide_xxxx.h5`: flattened MNIST pixels and 2-D coordinates for each bag.
- `mnist_fourbags.csv`: labels for the digit-8/9 counting task.
- `mnist_even_odd.csv`: labels for the even-versus-odd majority task.
- `mnist_adjacent_pairs.csv`: labels for the adjacent-pair detection task.
- `mnist_fourbags_plus.csv`: labels for the composite rule-based task.
- `evidence/<task>/slide_xxxx.pt`: per-instance evidence and digit identities saved
  for interpretability analyses.
- `images_shape.txt`: synthetic canvas sizes used when reconstructing spatial maps.
- `splits/<task>/`: cross-validation CSV files for every task.
- The generator balances the dataset for the requested task so that the least
  represented label still covers at least 25% of the slides.

## 2. Run the Bayes-MIL pipeline step by step

The following helpers mirror the manual commands usually issued against
`main.py`, `eval.py`, and the visualisation utilities. They are provided to keep
the workflow explicit and modular.

### 2.1 Train

```bash
python examples/mnist_train.py \
    --dataset-root /path/to/mnist_mil_dataset \
    --task mnist_fourbags \
    --exp-code mnist_demo \
    --k 5 --max-epochs 20 --lr 5e-4
```

All flags map one-to-one to the arguments consumed by `main.py`, so you can add
`--drop-out`, `--early-stopping`, or `--weighted-sample` as needed. The script
writes checkpoints under `<results-dir>/<exp-code>_s<seed>/`. Switch `--task` to
`mnist_even_odd`, `mnist_adjacent_pairs`, or `mnist_fourbags_plus` to train on the
other synthetic objectives without changing any other flags.

### 2.2 Evaluate

After training, evaluate the checkpoints with `eval.py` via:

```bash
python examples/mnist_evaluate.py \
    --dataset-root /path/to/mnist_mil_dataset \
    --results-dir results \
    --task mnist_fourbags \
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
    --task mnist_fourbags \
    --exp-code mnist_demo \
    --fold 0 --split test
```

The script writes PNGs to
`<dataset-root>/visualizations/<exp-code>_s<seed>/test_fold_<fold>/` and emits a
`heatmap_summary.csv` with predicted labels. Use `--split val` or `--split train`
to export other subsets, or `--split all` to process every slide contained in
the descriptor CSV. Pass `--skip-existing` to avoid re-rendering PNGs that are
already present. When switching between tasks, rerun the training and evaluation
helpers with the matching `--task` flag so that the checkpoint and split
metadata agree on the number of classes. The heatmap export utility validates
this and raises a clear error if a mismatch is detected.

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
    --task mnist_fourbags \
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
    --task mnist_fourbags \
    --model-type bmil-vis \
    --fold 0 --split test --all
```
