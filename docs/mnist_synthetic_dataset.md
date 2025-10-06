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

## 2. Launch training

Point `main.py` to the generated directory with the new `--task` options. Below
is an example for the binary classification task using Bayes-MIL-Vis.

```bash
python main.py \
    --data_root_dir /path/to/mnist_mil_dataset \
    --task mnist_binary \
    --model_type bmil-vis \
    --exp_code mnist_binary_example \
    --k 5 --max_epochs 20 --lr 5e-4
```

Swap `--task` to `mnist_ternary` to evaluate the three-class task. The script
reads the splits generated earlier (unless `--split_dir` overrides the path) and
loads features from `h5_files/`.

## 3. Visualize a synthetic slide

You can quickly inspect the generated bags with the helper below. It rebuilds
the MNIST grid stored in `h5_files/<slide>.h5` and saves a PNG preview that
annotates the slide-level labels.

```bash
python vis_utils/visualize_mnist_slide.py \
    --dataset-root /path/to/mnist_mil_dataset \
    --slide-id slide_0005 \
    --output preview.png
```

If `--output` is omitted the script drops images under
`<dataset-root>/visualizations/`.

## 4. Troubleshooting

- Ensure that `torchvision` is installed in the active environment so the script
  can download MNIST.
- When re-generating the dataset in the same directory, old files are overwritten.
- If you wish to experiment with custom label definitions, modify the helper
  script before generating the dataset.
