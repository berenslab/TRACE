import pandas as pd
from pathlib import Path

import argparse
from datetime import datetime
import lightning
import numpy as np
import os
import time
import torch
import tsimcne
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import sklearn
from sc_utils import (
    ContrastiveTrialPairGenerator,
    TimeSeriesDataset,
    TimeSeriesMLP,
    TimeSeriesProjectionHead,
    TorchVectorizedContrastiveTrialPairGenerator,
)
from timeseries_data import load_data_bc, load_data_sc, load_data_toy, load_data_rgc
from sc_utils import knn_accuracy, ari_score
from tsimcne.losses import infonce


class C4tsimcne(ContrastiveTrialPairGenerator):
    def __getitem__(self, idx):
        sample1, sample2 = super().__getitem__(idx)
        x = torch.vstack(
            (torch.from_numpy(sample1), torch.from_numpy(sample2))
        )
        dummy_label = 1
        return x, dummy_label


class C4tsimcneSingle(TimeSeriesDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        dummy_label = 1
        return sample, dummy_label


class NeuroDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        data,
        n_trials_pp,
        batch_size=2**13,
        num_workers=16,
        data_aug=False,
        noise_samples=None,
        device="cuda",
        seed=0,
        **kwargs,
    ):
        super().__init__()
        self.data = data
        self.n_trials_pp = n_trials_pp
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_aug = data_aug
        self.noise_samples = noise_samples
        self.device = device
        self.seed = seed
        self.kwargs = kwargs

    @staticmethod
    def collate_fn(data):
        b1, lbl1 = list(zip(*data))
        im1, im2 = list(zip(*b1))
        return torch.vstack(
            (torch.stack(im1), torch.stack(im2))
        ), torch.tensor(lbl1)

    def train_dataloader(self):
        #dataset = C4tsimcne(
        #    self.data,
        #    self.n_trials_pp,
        #    data_aug=self.data_aug,
        #    noise_samples=self.noise_samples,
        #)
        #return torch.utils.data.DataLoader(
        #    dataset,
        #    batch_size=self.batch_size,
        #    num_workers=self.num_workers,
        #    shuffle=True,
        #    collate_fn=self.collate_fn,
        #    **self.kwargs,
        #)
        return TorchVectorizedContrastiveTrialPairGenerator(
            trials = self.data,
            n_trials_pp = self.n_trials_pp,
            batch_size = self.batch_size,
            data_aug = self.data_aug,
            noise_samples = self.noise_samples,
            shuffle=True,
            drop_last = True,
            seed = self.seed,
            device=self.device
        )

    def predict_dataloader(self):
        kwargs = self.kwargs.copy()
        kwargs.pop("drop_last", None)
        dataset = C4tsimcneSingle(self.data)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            **kwargs,
        )

    def val_dataloader(self):
        return [self.train_dataloader(), self.predict_dataloader()]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_name", type=str, default="TimeSeriesMLP"
    )
    parser.add_argument(
        "-ds",
        "--dataset_name",
        type=str,
        default="sc",  # "bc",
        help="Name of the data set.",
    )
    parser.add_argument(
        "-r",
        "--run",
        type=int,
        help="Seed for random number generator.",
        default=42,
    )
    parser.add_argument(
        "-e", "--epochs", default=200, type=int, help="Number of epochs."
    )
    parser.add_argument(
        "-b", "--batch_size", default=1024, type=int, help="Batch size."
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=lambda x: None if x.lower() == 'none' else float(x),
        default=None,
        help="Learning rate (float or None).",
    )
    parser.add_argument(
        "-opt",
        "--optimizer",
        type=str,
        choices=['sgd', 'adam', 'adamw'],
        default='sgd',
        help="Optimizer type.",
    )
    parser.add_argument(
        "-met",
        "--metric",
        type=str,
        choices=['euclidean', 'cosine', 'gauss'],
        default='euclidean',
        help="Optimizer type.",
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=Path,
        default="",
        help="Directory to save results."
    )
    parser.add_argument(
        "-o",
        "--output_dim",
        type=int,
        help="Dimensions of final layer.",
        default=2,
    )
    parser.add_argument(
        "-tpp",
        "--n_trials_pp",
        type=int,
        nargs='+',  # Accept one or more integers
        default=[7, 5],
        help="Number of trials per pair to average over. Provide one or two integers (e.g., '7 5')."
    )
    parser.add_argument(
        "-a",
        "--augmentations",
        action="store_true",
        help="Use common data augmentations instead of partial means.",
    )
    parser.add_argument(
        "-fb",
        "--flatten_bar",
        action="store_true",
        help="Flatten responses 8 directions or use mean across directions.",
    )
    parser.add_argument(
        "-dev",
        "--device",
        type=str,
        default="cuda",
        help="Device used for computing positive pairs, either 'cpu' or 'cuda'.",
    )
    args = parser.parse_args()

    # Set parameters
    model_name = str(args.model_name)
    torch.manual_seed(args.run)
    num_workers = 32
    if args.learning_rate is None:
        lr = 'auto_batch'
    else:
        lr = args.learning_rate
    if len(args.n_trials_pp) == 1:
        formatted_n_trials_pp = f"{args.n_trials_pp[0]}"
    else:
        formatted_n_trials_pp = f"{args.n_trials_pp[0]}_{args.n_trials_pp[1]}"

    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime("%Y%m%d_%H%M%S")
    file_name = (
        f"{datetime_string}"
        f"_embd_{model_name}"
        f"_dataset{args.dataset_name}"
        f"_epochs{args.epochs}"
        f"_lr{lr}"
        f"_batchsize{args.batch_size}"
        f"_outputdim{args.output_dim}"
        f"_run{args.run}"
        f"_ntrialpp{formatted_n_trials_pp}"
        f"_flattenbar{str(args.flatten_bar)}"
        f"_dataug{args.augmentations}"
        f"_device{args.device}"
    )
    print(f"Directory: {args.dir}")
    print(f"File name: {file_name}")
    plots_dir = args.dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.dir
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.dataset_name == "sc":
        #data_chirp, data_bar, labels, type_names = load_data_sc(flatten_bar=args.flatten_bar)
        data_chirp, data_bar = load_data_sc(flatten_bar=args.flatten_bar)
        data = [data_chirp, data_bar]
        if args.augmentations:
            if args.flatten_bar:
                noise_samples = np.load(
                    "/gpfs01/berens/data/data/superior_colliculus/"
                    "noise_samples_flattenbarTrue.npy"
                )
            else:
                noise_samples = np.load(
                    "/gpfs01/berens/data/data/superior_colliculus/"
                    "noise_samples.npy"
                )
        else:
            noise_samples = None
    
    elif args.dataset_name == "bc":
        d1, labels, type_names = load_data_bc()
        data = [d1]
        if args.augmentations:
            noise_samples = np.load(
                "/gpfs01/euler/User/dgonschorek/projects/_3_Postdoc/TRACE_Contrastive_Learning/TRACE/data/BC/"
                "noise_samples_local_chirp.npy"
            )
        else:
            noise_samples = None
    
    elif args.dataset_name == "rgc":
        #data_chirp, data_bar, labels = load_data_rgc()
        #data = [data_chirp, data_bar]
        data_chirp, labels = load_data_rgc()
        data_chirp = data_chirp.astype(np.float32)
        data = [data_chirp]
        if args.augmentations:
            #noise_samples = np.load(
            #    "/gpfs01/euler/User/dgonschorek/projects/_3_Postdoc/All_RGCs/data/TRACE/"
            #    "noise_samples_rgc.npy"
            #)
            noise_samples = np.load(
                "/gpfs01/euler/User/dgonschorek/projects/_3_Postdoc/All_RGCs/data/TRACE/"
                "noise_samples_rgc_chirp.npy"
            )
        else:
            noise_samples = None
        
    elif args.dataset_name == "toy":
        d1, labels, type_names = load_data_toy()
        data = [d1]
        if args.augmentations:
            noise_samples = np.load(
                "/gpfs01/berens/user/lschmors/Code/superior_colliculus"
                "/20241211_simple_toy_dataset/data/toy_data_noise_samples.npy"
            )
        else:
            noise_samples = None

    dm = NeuroDataModule(
        data,
        n_trials_pp=args.n_trials_pp,
        data_aug=args.augmentations,
        noise_samples=noise_samples,
        batch_size=(bs := args.batch_size),
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=True,
        device=args.device
    )

    # Initialize model
    mod_neuro = tsimcne.PLtSimCNE(
        backbone=TimeSeriesMLP(
            input_features=sum(ds.shape[2] for ds in data),
            n_features=(bbdim := 128),
        ),
        backbone_dim=bbdim,
        projection_head=TimeSeriesProjectionHead(n_input=bbdim, n_output=args.output_dim),
        out_dim=args.output_dim,
        anneal_to_dim=args.output_dim,
        n_epochs=(n := args.epochs),
        batch_size=bs,
        warmup_epochs=0 if n < 99 else 10,
        loss=infonce.InfoNCET(dof=1),
        dof=1,
        eval_ann=False,
        lr=lr,
        optimizer_name=args.optimizer,
        metric=args.metric,
        temperature=0.5,

    )

    trainer = lightning.Trainer(
        max_epochs=n, check_val_every_n_epoch=20, log_every_n_steps=5
    )

    # Fit model
    start = time.time()
    trainer.fit(mod_neuro, datamodule=dm)
    train_time = time.time() - start
    mod_neuro.train_time = train_time
    print(f"Training time: {mod_neuro.train_time:.2f} sec")

    # Get embedding
    out = trainer.predict(mod_neuro, datamodule=dm)
    Z = torch.vstack([x[0] for x in out])
    print(f"Embedding shape: {Z.shape}")

    # Save embedding
    embd_filepath = os.path.join(models_dir, f"{file_name}.npy")
    np.save(embd_filepath, Z)

    # Evaluate embedding
    n_neighbors = 15
    mod_neuro.knn_acc = knn_accuracy(embedding=Z, labels=labels, n_neighbors=n_neighbors)
    mod_neuro.ari_score = ari_score(embedding=Z, true_labels=labels)

    # Save loss
    loss_df = pd.read_csv(os.path.join(mod_neuro.logger.log_dir, 'metrics.csv'))
    mod_neuro.loss_df = loss_df

    # Save model
    mod_neuro.rng = None
    model_filepath = os.path.join(models_dir, f"{file_name}.pth")
    torch.save(dict(model=mod_neuro, sd=mod_neuro.state_dict()), model_filepath)
    print(f"Model saved as: {model_filepath}")

    # Save metrics
    save_metrics = pd.DataFrame({
        'datetime_string': [datetime_string],
        'ARI': [mod_neuro.ari_score],
        'kNN_acc': [mod_neuro.knn_acc],
        'loss': [mod_neuro.loss_df.groupby('epoch')['loss'].mean().values[-1]],
        'train_time': [mod_neuro.train_time],
    })
    save_metrics.to_csv(Path(args.dir / "grid_search") / (f"{datetime_string}_{args.dataset_name}"
                                                        f"_outputdim{args.output_dim}"
                                                        f"_ntrialpp{formatted_n_trials_pp}"
                                                        f"_flattenbar{str(args.flatten_bar)}"
                                                        f"_run{args.run}"
                                                        f"_dataug{args.augmentations}"
                                                        f"_batchsize{args.batch_size}"
                                                        f"_epochs{args.epochs}"
                                                        f"_lr{mod_neuro.lr}"
                                                        f".csv"),
                        index=False)

    # Plot embedding
    style_file = Path.home() / "berenslab.mplstyle"
    if style_file.exists():
        plt.style.use(style_file)
    fig, ax = plt.subplots(figsize=(3, 3))
    cmap = ListedColormap(sns.husl_palette(np.unique(labels).shape[0]).as_hex())
    if Z.shape[1] == 2:
        ax.scatter(*Z.T, c=labels, cmap=cmap, alpha=1, s=2)
    elif Z.shape[1] > 2:
        # Apply PCA
        pca = sklearn.decomposition.PCA(n_components=2)
        X_pca = pca.fit_transform(Z)
        # Plot
        ax.scatter(*X_pca.T, c=labels, cmap=cmap, alpha=1, s=2)
    # Save figure
    print(Path(plots_dir) / f"{file_name}.png")
    fig.savefig(Path(plots_dir) / f"{file_name}.png")
    plt.close(fig)

    # Plot loss
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    loss_by_epoch = mod_neuro.loss_df.groupby('epoch')['loss'].mean()
    loss_by_epoch_std = mod_neuro.loss_df.groupby('epoch')['loss'].std()
    ax.plot(loss_by_epoch.index, loss_by_epoch.values, 'k')
    ax.fill_between(loss_by_epoch.index,
                    loss_by_epoch.values - loss_by_epoch_std.values,
                    loss_by_epoch.values + loss_by_epoch_std.values,
                    facecolor='gray',
                    edgecolor='none',
                    alpha=0.5)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    # Save figure
    print(Path(plots_dir) / f"{file_name}.png")
    fig.savefig(Path(plots_dir) / f"{file_name}_loss.png")
    plt.close(fig)

if __name__ == "__main__":
    main()
