# TRACE: Time series Representation Analysis through Contrastive Embeddings

This is the package repository for TRACE, a contrastive learning framework that creates interpretable 2D embeddings of high-dimensional time series data by generating positive pairs through trial averaging, exploiting the multi-trial structure common in neuroscience experiments.

# Installation

TRACE uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the TRACE repository:
```bash
git clone https://github.com/berenslab/TRACE.git
cd TRACE
```

3. Set up the environment:
```bash
uv sync
```
→ This will install all dependencies pinned in `uv.lock` and install the TRACE package itself in editable mode.

4. Verify the install:
```bash
uv run trace --help
```
→ This should show the argparse help. If you get command not found, run `uv sync` again.

# Usage

Run TRACE with default settings:

```bash
uv run trace -d path/to/output/directory
```

### Required Arguments
- `-d, --dir`: Directory to save results and trained models

### Data Arguments
- `-ds, --dataset_name`: Name of the dataset (default: `sc` for the superior colliculus dataset)
- `-pa, --path_to_augmented_data`: Path to pre-computed augmented data (optional)

### Model Arguments
- `-m, --model_name`: Model architecture (default: `TimeSeriesMLP`)
- `-o, --output_dim`: Dimensions of the output embedding (default: `2`)
- `-met, --metric`: Distance metric (`euclidean`, `cosine`, `gauss`; default: `euclidean`)

### Training Arguments
- `-e, --epochs`: Number of training epochs (default: `200`)
- `-b, --batch_size`: Batch size (default: `1024`)
- `-lr, --learning_rate`: Learning rate (default: `None` for auto-scaling with batch size)
- `-opt, --optimizer`: Optimizer type (`sgd`, `adam`, `adamw`; default: `sgd`)
- `-r, --run`: Random seed (default: `42`)

### TRACE-Specific Arguments
- `-tpp, --n_trials_pp`: Number of trials to average per positive pair. Provide one or two integers (default: `7 5`)
  - Single value: uses same number for both views
  - Two values: different numbers for each view
- `-a, --augmentations`: Use standard data augmentations instead of trial averaging (flag)
- `-fb, --flatten_bar`: Flatten responses across 8 directions vs. use mean (flag)

### System Arguments
- `-dev, --device`: Computing device (`cpu` or `cuda`; default: `cuda`)

## Example Commands

TRACE can be used with specified hyperparameters, custom datasets, and with either trial averaging to generate positive pairs or with standard data augmentations. Examples follow below:

**Train with custom hyperparameters:**
```bash
uv run trace -d results/ -e 1000 -b 1280 -lr 0.1
```

**Use custom data:**
```bash
uv run trace -d results/ -pd data/neural_recordings.npy -pl data/labels.npy
```

**Adjust trial averaging:**
```bash
uv run trace -d results/ -tpp 7 5  # 10 trials for first stimulus, 8 for second
```

**Use standard augmentations instead of trial averaging:**
```bash
uv run trace -d results/ -a
```

# Contributions
* Lisa Schmors (Maintainer)
* Dominic Gonschorek
* Jan Niklas Böhm
* Sebastian Damrich

# Citation
```
@article{schmors2025trace,
      title={TRACE: Contrastive learning for multi-trial time-series data in neuroscience},
      author={Schmors, Lisa and Gonschorek, Dominic and B{\"o}hm, Jan Niklas and Qiu, Yongrong and Zhou, Na and Kobak, Dmitry and Tolias, Andreas and Sinz, Fabian and Reimer, Jacob and Franke, Katrin and others},
      journal={Neural Information Processing Systems 2025},
      year={2025}
}
```