import os

import lightning
import numpy as np
import torch
import tsimcne
from tsimcne.losses import infonce

from trace_neuro.datasets import NeuroDataModule
from trace_neuro.models import TimeSeriesMLP, TimeSeriesProjectionHead

class TRACE:
    """Time series representation analysis through contrastive embeddings.

    Parameters
    ----------
    output_dim : int
        Dimensionality of the learned embedding.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Number of positive pairs per training batch.
    lr : float or None
        Learning rate. ``None`` scales automatically with batch size
        (0.03 * batch_size / 256).
    optimizer : {"sgd", "adam", "adamw"}
        Optimizer.
    metric : {"euclidean", "cosine", "gauss"}
        Metric used in the contrastive loss.
    backbone : "mlp" or nn.Module
        Encoder backbone. "mlp" instantiates :class:`TimeSeriesMLP`
        with a 128-d output; passing an ``nn.Module`` uses it directly
    n_trials_pp : int or list of int
        Number of trials averaged per positive pair.
        If list, [n1, n2], different counts are used to average across per stimulus.
    temperature : float
        InfoNCE temperature.
    augmentations : bool
        Use stochastic augmentations instead of trial averaging.
        Requires to compute or pre-computed noise_samples.
    noise_samples : np.ndarray or None
        Pre-computed noise for the noise augmentation, shape
        (n_noise, time).  Required when augmentations=True.
    device : str
        "cuda" or "cpu".
    random_state : int
        Seed for reproducibility.

    Notes
    -----
    TRACE does not normalise input data.  Pass z-scored responses
    (per-neuron) to both fit() and transform() for consistent embeddings.
    """

    def __init__(
        self,
        output_dim=2,
        n_epochs=200,
        batch_size=1024,
        lr=None,
        optimizer="sgd",
        metric="euclidean",
        backbone="mlp",
        n_trials_pp=(7, 5),
        temperature=0.5,
        augmentations=False,
        noise_samples=None,
        device="cuda",
        random_state=42,
    ):
        self.output_dim = output_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.metric = metric
        self.backbone = backbone
        self.n_trials_pp = (
            list(n_trials_pp) if hasattr(n_trials_pp, "__iter__") else [n_trials_pp]
        )
        self.temperature = temperature
        self.augmentations = augmentations
        self.noise_samples = noise_samples
        self.device = device
        self.random_state = random_state

    def _resolved_lr(self):
        return "auto_batch" if self.lr is None else self.lr

    def _make_datamodule(self, X):
        data = X if isinstance(X, list) else [X]
        num_workers = min(4, os.cpu_count() or 1)
        return NeuroDataModule(
            data,
            n_trials_pp=self.n_trials_pp,
            data_aug=self.augmentations,
            noise_samples=self.noise_samples,
            batch_size=self.batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            drop_last=True,
            device=self.device,
        )

    def _build_model(self, input_dim):
        if self.backbone == "mlp":
            bb = TimeSeriesMLP(input_features=input_dim, out_dim=128)
            bb_dim = 128
        else:
            bb = self.backbone
            with torch.no_grad():
                dummy = torch.zeros(1, input_dim)
                bb_dim = bb(dummy).shape[-1]

        head = TimeSeriesProjectionHead(n_input=bb_dim, n_output=self.output_dim)
        n = self.n_epochs
        return tsimcne.PLtSimCNE(
            backbone=bb,
            backbone_dim=bb_dim,
            projection_head=head,
            out_dim=self.output_dim,
            anneal_to_dim=self.output_dim,
            n_epochs=n,
            batch_size=self.batch_size,
            warmup_epochs=0 if n < 99 else 10,
            loss=infonce.InfoNCET(dof=1),
            dof=1,
            eval_ann=False,
            lr=self._resolved_lr(),
            optimizer_name=self.optimizer,
            metric=self.metric,
            temperature=self.temperature,
        )

    def fit(self, X):
        """Train on ``X``.

        Parameters
        ----------
        X : np.ndarray or list of np.ndarray
            Neural responses, shape ``(n_neurons, n_trials, n_time)``.
            Pass a list if your have responses for multiple stimuli with different trial counts.

        Returns
        -------
        self
        """
        torch.manual_seed(self.random_state)
        data = X if isinstance(X, list) else [X]
        self.input_dim_ = sum(ds.shape[2] for ds in data)

        dm = self._make_datamodule(X)
        self.model_ = self._build_model(self.input_dim_)

        accelerator = "gpu" if self.device.startswith("cuda") else "cpu"
        self.trainer_ = lightning.Trainer(
            max_epochs=self.n_epochs,
            accelerator=accelerator,
            devices=1,
            check_val_every_n_epoch=min(20, self.n_epochs),
            log_every_n_steps=5,
        )
        self.trainer_.fit(self.model_, datamodule=dm)
        return self

    def transform(self, X):
        """Embed ``X`` with the trained backbone.

        Works for training neurons and held-out neurons: the backbone is an
        inductive map from response profile to embedding space, so any
        neuron recorded under the same stimulus protocol (same ``n_time``)
        can be embedded without retraining.  Normalise ``X`` the same
        way as the training data before calling this method.

        Parameters
        ----------
        X : np.ndarray or list of np.ndarray
            Neural responses, shape ``(n_neurons, n_trials, n_time)``.

        Returns
        -------
        Z : np.ndarray, shape ``(n_neurons, output_dim)``
        """
        dm = self._make_datamodule(X)
        out = self.trainer_.predict(self.model_, datamodule=dm)
        return torch.vstack([x[0] for x in out]).detach().cpu().numpy()

    def fit_transform(self, X):
        """Fit on ``X`` and return its embedding.

        Parameters
        ----------
        X : np.ndarray or list of np.ndarray

        Returns
        -------
        Z : np.ndarray, shape ``(n_neurons, output_dim)``
        """
        return self.fit(X).transform(X)