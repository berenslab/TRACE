# TRACE: Time series Representation Analysis through Contrastive Embeddings

This is the package repository for TRACE, a contrastive learning framework that creates interpretable 2D embeddings of high-dimensional time series data by generating positive pairs through trial averaging, exploiting the multi-trial structure common in neuroscience experiments.

# Usage

Run TRACE with default settings:
```bash
python tsimcne_run.py -d path/to/output/directory
```

### Required Arguments
- `-d, --dir`: Directory to save results and trained models

### Data Arguments
- `-ds, --dataset_name`: Name of the dataset (default: `sc`)
- `-pd, --path_to_data`: Path to the data file
- `-pl, --path_to_labels`: Path to the labels file
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
python tsimcne_run.py -d results/ -e 1000 -b 1280 -lr 0.1
```

**Use custom data:**
```bash
python tsimcne_run.py -d results/ -pd data/neural_recordings.npy -pl data/labels.npy
```

**Adjust trial averaging:**
```bash
python tsimcne_run.py -d results/ -tpp 10 8  # 10 trials for first view, 8 for second
```

**Use standard augmentations instead of trial averaging:**
```bash
python tsimcne_run.py -d results/ -a
```

# Contributions
* Lisa Schmors (Maintainer)
* Dominic Gonschorek
* Jan Niklas Böhm
* Sebastian Damrich

# Citation
If you find the code useful for your research, please consider citing our work:
````
@misc{schmors2025trace,
      title={TRACE: Contrastive learning for multi-trial time-series data in neuroscience},
      author={Schmors, Lisa and Gonschorek, Dominic and B{\"o}hm, Jan Niklas and Qiu, Yongrong and Zhou, Na and Kobak, Dmitry and Tolias, Andreas and Sinz, Fabian and Reimer, Jacob and Franke, Katrin and others},
      journal={arXiv preprint arXiv:2506.04906},
      year={2025}
}
````
