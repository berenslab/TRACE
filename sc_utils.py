import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data_aug import AmpJitter, Noise, TempJitter


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data_chirp,
        data_bar,
        labels=None,
        transform=None,
        noise_samples=None,
    ):
        """
        Args:
            data_chirp (np.array):
                Responses to chirp stimulus.
                Time series data of shape (num_samples, num_trials, num_features).
            data_bar (np.array):
                Responses to moving bar stimulus.
                Time series data of shape (num_samples, num_trials, num_features).
            labels (np.array):
                Labels for the data retrieved from clustering.
            transform (callable, optional):
                Optional transform to be applied on a sample instead of generating pairs
                using random sub-samples of trials.
        """
        self.data_chirp = data_chirp
        self.data_bar = data_bar
        self.labels = labels
        self.transform = transform
        self.noise_samples = noise_samples

    def __len__(self):
        return self.data_chirp.shape[0]

    def __getitem__(self, idx):
        sample_chirp = np.mean(self.data_chirp[idx], axis=0)  # Compute mean over trials
        sample_bar = np.mean(self.data_bar[idx], axis=0)  # Compute mean over trials
        sample = np.concatenate([sample_chirp, sample_bar])
        if self.transform:
            sample = self.transform(sample)
        if self.labels is not None:
            return sample, self.labels[idx]
        return sample


class ContrastiveTrialPairGenerator(Dataset):
    """
    Generates pairs of samples for contrastive learning.
    Supports both dynamic trial sampling and data augmentation.
    For the dynamic trial sampling, each pair is newly computed by selecting random trials
    for mean computation.
    """
    def __init__(self, dataset_chirp, dataset_bar,
                 n_trials_pos_pair_chirp=7,
                 n_trials_pos_pair_bar=5,
                 data_aug=False,
                 noise_samples=None,
                 ):
        """
        Args:
            dataset_chirp (np.ndarray): Chirp stimulus responses
            dataset_bar (np.ndarray): Bar stimulus responses
            n_trials_pos_pair_chirp (int): Trials to average for chirp positive pairs
            n_trials_pos_pair_bar (int): Trials to average for bar positive pairs
            data_aug (bool): Whether to use data augmentations
            noise_samples (np.ndarray): precomputed noise samples based on the
        """
        # Chirp
        self.dataset_chirp = dataset_chirp
        self.num_trials_chirp = dataset_chirp.shape[1]  # number of trials recorded
        self.ntrials_chirp = n_trials_pos_pair_chirp  # trials to average over for pos. pair

        # Moving bar
        self.dataset_bar = dataset_bar
        self.num_trials_bar = dataset_bar.shape[1]
        self.ntrials_bar = n_trials_pos_pair_bar

        # Apply data augmentations True/False
        if data_aug:
            assert noise_samples is not None, 'Please provide noise for noise augmentation'
        self.data_aug = data_aug
        self.noise_samples = noise_samples

    def __len__(self):
        return self.dataset_chirp.shape[0]

    def __getitem__(self, idx):
        if self.data_aug:
            # Use data augmentations to generate positive pair
            self.transform = get_transforms(noise_samples=self.noise_samples)

            item_chirp = np.mean(self.dataset_chirp[idx, :, :], axis=0)
            item_bar = np.mean(self.dataset_bar[idx, :, :], axis=0)
            item = np.concatenate([item_chirp, item_bar])

            sample1 = self.transform(item)
            sample2 = self.transform(item)
        else:
            # Use random sub-sample of trials to generate positive pair
            assert ((self.ntrials_chirp <= self.num_trials_chirp / 2) or
                    (self.ntrials_bar <= self.num_trials_bar / 2)), \
                'Not enough trials to average over for generating positive pair. ' \
                'Please choose a smaller n_trials_pos_pair'

            # Generate positive pair for chirp response
            n_pos_trials_chirp = self.ntrials_chirp
            num_trials_chirp = self.num_trials_chirp
            trials_indices1_chirp, trials_indices2_chirp = self._generate_dynamic_pairs_indices(
                n_pos_trials_chirp, num_trials_chirp)
            sample1_chirp = np.mean(self.dataset_chirp[idx, trials_indices1_chirp, :], axis=0)
            sample2_chirp = np.mean(self.dataset_chirp[idx, trials_indices2_chirp, :], axis=0)

            # Generate positive pair for moving bar responses
            n_pos_trials_bar = self.ntrials_bar
            num_trials_bar = self.num_trials_bar
            trials_indices1_bar, trials_indices2_bar = self._generate_dynamic_pairs_indices(
                n_pos_trials_bar, num_trials_bar)
            sample1_bar = np.mean(self.dataset_bar[idx, trials_indices1_bar, :], axis=0)
            sample2_bar = np.mean(self.dataset_bar[idx, trials_indices2_bar, :], axis=0)

            sample1 = np.concatenate([sample1_chirp, sample1_bar])
            sample2 = np.concatenate([sample2_chirp, sample2_bar])

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
        remaining_numbers = [
            num for num in all_indices if num != trials_indices1
        ]
        trials_indices2 = random.sample(remaining_numbers, n_pos_trials)
        return trials_indices1, trials_indices2


class TimeSeriesMLP(nn.Module):
    def __init__(self, input_features, n_features=2):
        super(TimeSeriesMLP, self).__init__()
        self.fc1 = nn.Linear(input_features, 768)  # First layer
        self.fc2 = nn.Linear(768, 512)  # Second layer
        self.fc3 = nn.Linear(512, 256)  # Third layer
        self.fc4 = nn.Linear(256, n_features)  # Last layer

    def forward(self, x):
        # Flatten the input in case it comes from a conv layer or it's not already flat
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def get_transforms(noise_samples):
    """
    Returns a list of data augmentations to be applied to the time series data.

    Args:
        noise_samples (np.array): pre-computed noise samples used for the Noise
        augmentation. Shape (num_samples, time_steps)

    Returns:
        transform (torchvision.transforms.Compose): Composed transformations.
    """

    # TODO: Add normalization?
    # normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose(
        [
            transforms.RandomApply([AmpJitter()], p=0.7),
            transforms.RandomApply([TempJitter()], p=0.6),
            transforms.RandomApply(
                [Noise(noise_samples=noise_samples)], p=0.5
            ),
            # normalize,
        ]
    )
    return transform


