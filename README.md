## Installation

Make sure you have `uv` [installed](https://docs.astral.sh/uv/getting-started/installation/).

## Usage

### Training

```bash
uv run train --data-dir <path-to-training-data>
```

Options:

- `--checkpoint-dir`: Directory to save checkpoints (default: `checkpoints/`)
- `--yolo-epochs`: YOLO training epochs (default: 100)
- `--regressor-epochs`: Regressor training epochs (default: 100)

### Prediction

```bash
uv run predict <image1> <image2> ...
```

Options:

- `--yolo-checkpoint`: Path to YOLO checkpoint (default: `checkpoints/yolo_best.pt`)
- `--regressor-checkpoint`: Path to regressor checkpoint (default: `checkpoints/regressor_best.pt`)
