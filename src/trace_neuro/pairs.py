import random
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from trace_neuro.augmentations import (
    get_transforms,
    get_torch_vectorized_transforms,
)

class ContrastiveTrialPairGenerator(Dataset):
    """
    Generates pairs of samples for contrastive learning.
    Supports both dynamic trial sampling and data augmentation.
    For the dynamic trial sampling, each pair is newly computed by selecting random trials
    for mean computation.
    """

    def __init__(
        self,
        trials,
        n_trials_pp=[7, 5],
        data_aug=False,
        noise_samples=None,
    ):
        """
        Args:
            dataset_chirp (np.ndarray): Chirp stimulus responses
            dataset_bar (np.ndarray): Bar stimulus responses
            n_trials_pp (list): Trials to average for chirp positive pairs and for bar
            positive pairs
            data_aug (bool): Whether to use data augmentations
            noise_samples (np.ndarray): precomputed noise samples based on the covariance matrix
        """
        if not isinstance(trials, list):
            self.trials = [trials]
        else:
            self.trials = trials

        if isinstance(n_trials_pp, list) and len(n_trials_pp) != len(
            self.trials
        ):
            warnings.warn(
                f"Got {len(self.trials)} datasets, but got {len(n_trials_pp)=}"
            )
        self.n_trials_pp = n_trials_pp[: len(self.trials)]

        # Apply data augmentations True/False
        if data_aug:
            assert (
                noise_samples is not None
            ), "Please provide noise for noise augmentation"
        self.data_aug = data_aug
        self.noise_samples = noise_samples

        if self.data_aug:
            self.trials_mean = [ds.mean(axis=1) for ds in self.trials]
        else:
            # Use random sub-sample of trials to generate positive pair
            assert all(
                n_trial_sample <= ds.shape[1] / 2
                for ds, n_trial_sample in zip(self.trials, self.n_trials_pp)
            ), (
                "Not enough trials to average over for generating positive pair. "
                "Please choose a smaller n_trials_pp"
            )
            self.samples1 = [[]] * len(self.trials)
            self.samples2 = [[]] * len(self.trials)

    def __len__(self):
        return self.trials[0].shape[0]

    def __getitem__(self, idx):
        if self.data_aug:
            # Use data augmentations to generate positive pair
            self.transform = get_transforms(noise_samples=self.noise_samples)

            items = [ds[idx] for ds in self.trials_mean]
            item = np.concatenate(items)

            sample1 = self.transform(item)
            sample2 = self.transform(item)
        else:
            for i, (ds, n_trial_pp) in enumerate(
                zip(self.trials, self.n_trials_pp)
            ):
                # Generate positive pair
                trial_indices1, trial_indices2 = (
                    self._generate_dynamic_pairs_indices(
                        n_trial_pp, ds.shape[1]
                    )
                )
                sample1_ = np.mean(ds[idx, trial_indices1, :], axis=0)
                self.samples1[i] = sample1_
                sample2_ = np.mean(ds[idx, trial_indices2, :], axis=0)
                self.samples2[i] = sample2_
            sample1 = np.concat(self.samples1)
            sample2 = np.concat(self.samples2)

        return sample1, sample2

    def _generate_dynamic_pairs_indices(self, n_pos_trials, num_trials):
        """
        Generates random indices for positive pairs.

        Args:
            n_pos_trials (int): Number of trials to average over for positive pair.
            num_trials (int): Total number of trials available.

        Returns:
            trials_indices1 (list): List of indices for first trial in positive pair.
            trials_indices2 (list): List of indices for second trial in positive
        """
        all_indices = list(range(num_trials))
        trials_indices1 = random.sample(all_indices, n_pos_trials)
        remaining_numbers = [num for num in all_indices if num != trials_indices1]
        #remaining_numbers = [num for num in all_indices if num not in trials_indices1] # account for list
        trials_indices2 = random.sample(remaining_numbers, n_pos_trials)
        return trials_indices1, trials_indices2


class TorchVectorizedContrastiveTrialPairGenerator:
    """
    Generates pairs of samples for contrastive learning.
    Supports both dynamic trial sampling and data augmentation.
    For the dynamic trial sampling, each pair is newly computed by selecting random trials
    for mean computation.
    """

    def __init__(
            self,
            trials,
            n_trials_pp=[7, 5],
            data_aug=False,
            noise_samples=None,
            batch_size=1024,
            shuffle=False,
            drop_last=False,
            seed=0,
            device="cpu"
    ):
        """
        Args:
            dataset_chirp (np.ndarray): Chirp stimulus responses
            dataset_bar (np.ndarray): Bar stimulus responses
            n_trials_pp (list): Trials to average for chirp positive pairs and for bar
            positive pairs
            data_aug (bool): Whether to use data augmentations
            noise_samples (np.ndarray): precomputed noise samples based on the covariance matrix
        """
        if not isinstance(trials, list):
            self.trials = [trials]
        else:
            self.trials = trials

        if isinstance(n_trials_pp, list) and len(n_trials_pp) != len(
                self.trials
        ):
            warnings.warn(
                f"Got {len(self.trials)} datasets, but got {len(n_trials_pp)=}"
            )
        self.n_trials_pp = n_trials_pp[: len(self.trials)]
        self.device = device
        self.trials = [torch.from_numpy(t).to(device=self.device, dtype=torch.float32) for t in self.trials]

        # Apply data augmentations True/False
        if data_aug:
            assert (
                    noise_samples is not None
            ), "Please provide noise for noise augmentation"
        self.data_aug = data_aug
        if self.data_aug:
            self.noise_samples = torch.from_numpy(noise_samples).to(device=self.device, dtype=torch.float32)
            self.trials_mean = [ds.mean(axis=1) for ds in self.trials]
            self.transform = get_torch_vectorized_transforms(noise_samples=self.noise_samples)
        else:
            # Use random sub-sample of trials to generate positive pair
            assert all(
                n_trial_sample <= ds.shape[1] / 2
                for ds, n_trial_sample in zip(self.trials, self.n_trials_pp)
            ), (
                "Not enough trials to average over for generating positive pair. "
                "Please choose a smaller n_trials_pp"
            )

            # n_time = sum(ds.shape[1] for ds in self.trials)
            # self.sample1 = np.empty(shape=n_time)
            # self.sample2 = np.empty(shape=n_time)
            self.samples1 = [[]] * len(self.trials)
            self.samples2 = [[]] * len(self.trials)
            # print(f"{len(self.samples1)=}, {len(self.trials)=}")


        self.batch_size = torch.tensor(batch_size, dtype=int).to(self.device)
        self.dataset_len = torch.tensor(self.trials[0].shape[0], device=self.device)

        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        torch.manual_seed(self.seed)

        # Calculate number of  batches
        n_batches = torch.div(self.dataset_len, self.batch_size, rounding_mode="floor")
        remainder = torch.remainder(self.dataset_len, self.batch_size)
        if remainder > 0 and not self.drop_last:
            n_batches += 1
        self.n_batches = n_batches

        self.n_trials = self.trials[0].shape[1]

    def __len__(self):
        return self.trials[0].shape[0]

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.device)
        else:
            self.indices = None
        self.i = torch.tensor(0, device=self.device)
        return self

    def _get_batch_idx(self):
        if (self.i > self.dataset_len - self.batch_size and self.drop_last) or self.i >= self.dataset_len:
            raise StopIteration

        start = self.i
        end = torch.minimum(self.i + self.batch_size, self.dataset_len)
        if self.indices is not None:
            batch_idx = self.indices[start:end]
        else:
            batch_idx = torch.arange(start, end, dtype=int, device=self.device)
        self.i += self.batch_size
        return batch_idx

    def __len__(self):
        return self.n_batches

    def __next__(self):
        # this retrieves full batches
        batch_idx = self._get_batch_idx()
        if self.data_aug:
            # Use data augmentations to generate positive pair
            items = [ds[batch_idx] for ds in self.trials_mean]
            item = torch.cat(items, dim=0)
            sample1 = self.transform(item)
            sample2 = self.transform(item)
        else:
            for i, (ds, n_trial_pp) in enumerate(
                    zip(self.trials, self.n_trials_pp)
            ):
                # print(f"{i=}")
                # Generate positive pair

                # Fast way of creating indices for the trials in the partial means. Random sample and reorder for
                # permuted trial indices
                rand_vals = torch.rand(self.batch_size, ds.shape[1], device=self.device)
                shuffled_trial_indices = torch.argsort(rand_vals, dim=1)

                # Split shuffled indices for first and second partial mean
                trial_indices1 = shuffled_trial_indices[:, :n_trial_pp].unsqueeze(-1)
                trial_indices2 = shuffled_trial_indices[:, n_trial_pp:2 * n_trial_pp].unsqueeze(-1)

                # Select data and compute mean
                ds_batch = ds[batch_idx]

                subset1 = torch.gather(ds_batch, 1, trial_indices1.expand(-1, -1, ds.shape[2]))
                sample1_ = subset1.mean(dim=1)
                subset2 = torch.gather(ds_batch, 1, trial_indices2.expand(-1, -1, ds.shape[2]))
                sample2_ = subset2.mean(dim=1)

                self.samples1[i] = sample1_
                self.samples2[i] = sample2_
            sample1 = torch.cat(self.samples1, dim=-1)
            sample2 = torch.cat(self.samples2, dim=-1)


        # concatenate and add dummy labels for tsimcne
        return torch.cat([sample1, sample2], dim=0), torch.ones(self.batch_size, device=self.device)