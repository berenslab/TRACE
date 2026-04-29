import lightning
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from trace_neuro.pairs import ContrastiveTrialPairGenerator

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        datasets,
        labels=None,
        transform=None,
        noise_samples=None,
    ):
        """
        Args:
            datasets (np.array or list):
                List of datasets, e.g. Responses to chirp stimulus and moving bar stimulus.
                Time series data of shape (num_samples, num_trials, num_features).
            labels (np.array):
                Labels for the data retrieved from clustering.
            transform (callable, optional):
                Optional transform to be applied on a sample instead of generating pairs
                using random sub-samples of trials.
        """
        if not isinstance(datasets, list):
            self.datasets = [datasets]
        else:
            self.datasets = datasets
        self.labels = labels
        self.transform = transform
        self.noise_samples = noise_samples

    def __len__(self):
        return self.datasets[0].shape[0]

    def __getitem__(self, idx):
        samples = [np.mean(ds[idx], axis=0) for ds in self.datasets]
        sample = np.concatenate(samples)
        if self.transform:
            sample = self.transform(sample)
        if self.labels is not None:
            return sample, self.labels[idx]
        return sample


class C4tsimcne(ContrastiveTrialPairGenerator):
    """
    Contrastive pair dataset adapted to the t-SimCNE training interface.

    Wrapper around :class:`ContrastiveTrialPairGenerator` that reshapes
    its output to match what :class:`tsimcne.PLtSimCNE` expects from a
    ``Dataset``: a ``(sample, label)`` tuple where ``sample`` carries both
    views of the positive pair stacked along the leading dimension.

    The parent class returns two NumPy arrays ``(sample1, sample2)`` of
    shape ``(n_features,)`` each. This wrapper stacks them into a single
    tensor of shape ``(2, n_features)`` and pairs it with a constant
    placeholder label, since the t-SimCNE training loop has a labelled
    ``Dataset`` signature even though contrastive training itself does not
    use labels. The two views are subsequently split apart in
    :meth:`NeuroDataModule.collate_fn` after batching.

    Parameters
    ----------
    Inherited from :class:`ContrastiveTrialPairGenerator`. See parent
    class for details on ``trials``, ``n_trials_pp``, ``data_aug``, and
    ``noise_samples``.

    Returns
    -------
    sample : torch.Tensor
        Tensor of shape ``(2, n_features)`` where ``sample[0]`` and
        ``sample[1]`` are the two views of the positive pair.
    dummy_label : int
        Constant ``1``, present only to satisfy the t-SimCNE
        ``(input, target)`` interface. Has no role in training.
    """
    def __getitem__(self, idx):
        sample1, sample2 = super().__getitem__(idx)
        x = torch.vstack(
            (torch.from_numpy(sample1), torch.from_numpy(sample2))
        )
        dummy_label = 1
        return x, dummy_label


class C4tsimcneSingle(TimeSeriesDataset):
    """
    Single-view dataset adapted to the t-SimCNE prediction interface.

    Thin wrapper around :class:`TimeSeriesDataset` used at inference time
    (i.e. inside :meth:`NeuroDataModule.predict_dataloader`) to extract
    embeddings for every sample in the dataset.

    Unlike :class:`C4tsimcne`, this wrapper does not generate positive
    pairs: it returns a single trial-averaged sample per index, identical
    to the parent class's output. The only modification is that the return
    is wrapped in a ``(sample, dummy_label)`` tuple so it conforms to the
    same ``(input, target)`` signature the t-SimCNE Lightning module
    expects from training-time datasets.

    Parameters
    ----------
    Inherited from :class:`TimeSeriesDataset`. See parent class for
    details on ``datasets``, ``labels``, ``transform``, and
    ``noise_samples``.

    Returns
    -------
    sample : np.ndarray
        Trial-averaged response of shape ``(n_features,)``, concatenated
        across stimuli if multiple datasets were provided.
    dummy_label : int
        Constant ``1``, present only to satisfy the t-SimCNE
        ``(input, target)`` interface. Has no role in inference.
    """
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        dummy_label = 1
        return sample, dummy_label


class NeuroDataModule(lightning.LightningDataModule):
    """
    PyTorch Lightning DataModule for TRACE contrastive training.

    Bundles the train and prediction dataloaders used by
    :class:`tsimcne.PLtSimCNE`. Training samples are positive pairs
    produced by :class:`C4tsimcne` (either via random trial-subset
    averaging or via stochastic augmentations of the trial mean); the
    prediction loader returns single trial-averaged samples via
    :class:`C4tsimcneSingle` and is used after training to extract the
    learned embedding for the full dataset.

    Parameters
    ----------
    data : np.ndarray or list of np.ndarray
        Multi-trial neural responses. Each array has shape
        ``(n_samples, n_trials, time)``.
        A list is used when there is a variable number of repeats
        in multiple stimuli (e.g. 15 repats for moving bar stimulus
        but only 10 repeats for chirp stimulus)
    n_trials_pp : list of int
        Number of trials averaged per view of a positive pair. A single
        value uses the same count for both views; two values
        (e.g. ``[7, 5]``) use different counts per view. When ``data``
        is a list, the list length must match the number of stimuli.
    batch_size : int
        Number of positive pairs per training batch.
    num_workers : int
        Number of worker processes for the DataLoaders.
    data_aug : bool, default=False
        If ``True``, generate positive pairs by applying stochastic
        augmentations (amplitude jitter, temporal jitter, noise)
        ``noise_samples`` must be provided in this case.
    noise_samples : np.ndarray, optional
        Pre-computed noise samples drawn from the temporal covariance
        matrix of the data, used by the noise augmentation. Shape
        ``(n_noise_samples, time)``. Required when
        ``data_aug=True``, ignored otherwise.
    device : {"cpu", "cuda"}, default="cuda"
        Device used by the (currently disabled) vectorized pair
        generator. Has no effect on the active per-sample path.
    seed : int, default=0
        Seed forwarded to the vectorized pair generator. Has no effect
        on the active per-sample path, which uses Python's ``random``
        module — seed that separately for full reproducibility.
    **kwargs
        Additional keyword arguments forwarded to ``DataLoader``
        (e.g. ``persistent_workers``, ``drop_last``).
    """
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
        dataset = C4tsimcne(
            self.data,
            self.n_trials_pp,
            data_aug=self.data_aug,
            noise_samples=self.noise_samples,
        )
        # Non vectorized version
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
            **self.kwargs,
        )
        # Un-comment for vectorized version
        #return TorchVectorizedContrastiveTrialPairGenerator(
        #    trials = self.data,
        #    n_trials_pp = self.n_trials_pp,
        #    batch_size = self.batch_size,
        #    data_aug = self.data_aug,
        #    noise_samples = self.noise_samples,
        #    shuffle=True,
        #    drop_last = True,
        #    seed = self.seed,
        #    device=self.device
        #)

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

