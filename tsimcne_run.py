from pathlib import Path

import lightning
import numpy as np
import torch
import tsimcne
from matplotlib import patheffects as path_effects
from matplotlib import pyplot as plt
from sc_utils import (
    ContrastiveTrialPairGenerator,
    TimeSeriesDataset,
    TimeSeriesMLP,
)
from timeseries_data import load_data_bc


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
        **kwargs,
    ):
        super().__init__()
        self.data = data
        self.n_trials_pp = n_trials_pp
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

    @staticmethod
    def collate_fn(data):
        b1, lbl1 = list(zip(*data))
        im1, im2 = list(zip(*b1))
        return torch.vstack(
            (torch.stack(im1), torch.stack(im2))
        ), torch.tensor(lbl1)

    def train_dataloader(self):
        dataset = C4tsimcne(
            self.data,
            self.n_trials_pp,
            # transform=self.transform,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
            **self.kwargs,
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
    d1, d2, labels = load_data_bc()
    data = [d1, d2]
    dm = NeuroDataModule(
        data,
        n_trials_pp=[5, 5],
        batch_size=(bs := 1024),
        num_workers=32,
        persistent_workers=True,
    )
    mod_neuro = tsimcne.PLtSimCNE(
        backbone=TimeSeriesMLP(
            input_features=sum(ds.shape[2] for ds in data),
            n_features=(bbdim := 512),
        ),
        backbone_dim=bbdim,
        n_epochs=(n := 1000),
        batch_size=bs,
        warmup_epochs=0 if n < 99 else 10,
        # loss=infonce.InfoNCET(dof=1),
        dof=1,
        anneal_to_dim=2,
        eval_ann=False,
    )
    trainer = lightning.Trainer(
        max_epochs=n, check_val_every_n_epoch=20, log_every_n_steps=5
    )
    trainer.fit(mod_neuro, datamodule=dm)
    out = trainer.predict(mod_neuro, datamodule=dm)
    Z = torch.vstack([x[0] for x in out])
    plt.style.use(Path.home() / "berenslab.mplstyle")
    fig, ax = plt.subplots(figsize=(3, 3))
    cmap = plt.get_cmap("tab20")
    ax.scatter(*Z.T, c=labels, cmap=cmap, alpha=1, s=2)
    type_names = [
        "1",
        "2",
        "3a",
        "3b",
        "4",
        "5t",
        "5o",
        "5i",
        "X",
        "6",
        "7",
        "8",
        "9",
        "R",
    ]

    def f_pe(c):
        return [path_effects.withStroke(linewidth=2, foreground=c, alpha=0.5)]

    [
        ax.text(
            *np.median(Z[labels == i], axis=0),
            lbl,
            path_effects=f_pe(cmap(i)),
        )
        for i, lbl in enumerate(type_names)
    ]
    fig.savefig(Path.home() / "dev/berenslab/neuro" / "nknipsel.png")
    # _ = await bot.send_photo(chat_id=tdict["chat_id"], photo="nknipsel.png")


if __name__ == "__main__":
    main()
