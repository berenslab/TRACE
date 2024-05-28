import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        """
        Args:
            data (np.array): Time series data of shape (num_samples, num_trials, num_features).
            labels (np.array): Labels for the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = np.mean(self.data[idx], axis=0)  # Compute mean over trials
        if self.transform:
            sample = self.transform(sample)
        if self.labels is not None:
            return sample, self.labels[idx]
        return sample

class ContrastiveTrialPairGenerator(Dataset):
    """
    Dynamically generates pairs of samples as positive pairs for contrastive learning.
    Each pair is freshly computed by selecting random trials for mean computation.
    """
    def __init__(self, dataset, transform=None, n_trials_pos_pair=5):
        self.dataset = dataset
        self.num_trials = dataset.shape[1] # number of trials recorded
        self.ntrials = n_trials_pos_pair # number of trials to average over to generate positive pair
        assert self.ntrials <= self.num_trials / 2, 'Not enough trials to average over for generating positive pair. Please choose a smaller n_trials_pos_pair'
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        trials_indices1, trials_indices2 = self._generate_dynamic_pairs_indices()

        sample1 = np.mean(self.dataset[idx, trials_indices1, :], axis=0)
        sample2 = np.mean(self.dataset[idx, trials_indices2, :], axis=0)

        if self.transform:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        return sample1, sample2

    def _generate_dynamic_pairs_indices(self):
        all_indices = list(range(self.num_trials))
        trials_indices1 = random.sample(all_indices, self.ntrials)
        remaining_numbers = [num for num in all_indices if num != trials_indices1]
        trials_indices2 = random.sample(remaining_numbers, self.ntrials)
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

"""
class TimeSeriesCNN(nn.Module):
    def __init__(self, input_channels=1, n_features=128):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        # Adjust the input features to the fc1 layer to match the output from the prints
        self.fc1 = nn.Linear(8320, n_features)  # Adjusted to match flattened size

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)  # Adds a channel dimension if not present
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
"""
