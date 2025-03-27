import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_aug import AmpJitter, Noise, TempJitter
from sklearn import neighbors, model_selection, metrics, mixture
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
        remaining_numbers = [
            num for num in all_indices if num != trials_indices1
        ]
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


class TorchVectorizedAmpJitter(object):
    """
    Amplitude jittering of the sample by scaling the amplitude.

    Args:
        lo (float): lower bound of the scaling factor
        hi (float): upper bound of the scaling factor

    Returns:
        sample_transformed (np.array): sample with scaled amplitude
    """
    def __init__(self, lo=.7, hi=1.3):

        self.lo = lo
        self.hi = hi

    def __call__(self, batch):
        amp_jit_values = torch.rand(batch.shape[0], device=batch.device, dtype=batch.dtype) * (self.hi - self.lo) + self.lo
        batch_transformed = batch * amp_jit_values.unsqueeze(-1)

        return batch_transformed

class TorchVectorizedTempJitter(object):
    """
    Temporal jittering of the sample by shifting the time axis.

    Args:
        shift_n_bins (int): number of bins to shift the sample

    Returns:
        sample_transformed (np.array): sample with shifted time axis
    """
    def __init__(self, shift_n_bins = 3):
        self.shift_n_bins = shift_n_bins

    def __call__(self, batch):
        # Generate the shift
        shifts_ = ((2 * torch.randint(low=0, high=2, size=(batch.shape[0],), device=batch.device) - 1) * torch.rand(size=(batch.shape[0],), device=batch.device) * self.shift_n_bins)
        # Calculate integer shift value
        int_shifts = torch.where(shifts_ >= 0, torch.ceil(shifts_), torch.floor(shifts_)).to(torch.int)

        #Apply the shift without padding first
        seq_len = batch.size(1)

        # Create an index tensor for the original positions
        index = torch.arange(seq_len, device=batch.device).repeat(batch.size(0), 1)

        # Compute shifted indices with wrap-around
        shifted_index = (index - int_shifts.unsqueeze(-1)) % seq_len

        # Gather the values based on the shifted indices
        rolled_batch = torch.gather(batch, 1, shifted_index)
        return rolled_batch


class TorchVectorizedNoise(object):
    """
    Add Gaussian noise to the sample based on a temporal covariance matrix. Noise samples
    have been pre-computed np.random.multivariate_normal(mean=np.zeros(cov_matrix.shape[0]), cov=cov_matrix)

    Args:
        noise_scale (float): scale of the noise

    Returns:[
        sample_transformed (np.array): sample with added noise
    """
    def __init__(self, noise_scale=.5, noise_samples=None):
        self.noise_scale = noise_scale
        self.noise_samples = noise_samples

    def __call__(self, batch):
        # Generate Gaussian noise based on the temporal covariance matrix

        # Randomly select one of the pre-computed noise samples
        noise_idx = torch.randint(0, self.noise_samples.shape[0], device=batch.device, size=(batch.shape[0],))
        noise = self.noise_samples[noise_idx]

        # Scale the noise and add to the original sample
        batch_noised = batch + self.noise_scale * noise

        return batch_noised

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

class TimeSeriesProjectionHead(nn.Module):
    def __init__(self, n_input, n_output=2):
        super().__init__()
        self.fc1 = nn.Linear(n_input, 1024)
        self.out = nn.Linear(1024, n_output)

    def forward(self, x):
        # Flatten the input in case it comes from a conv layer or it's not already flat
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
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

def get_torch_vectorized_transforms(noise_samples):
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
            transforms.RandomApply([TorchVectorizedAmpJitter()], p=0.7),
            transforms.RandomApply([TorchVectorizedTempJitter()], p=0.6),
            transforms.RandomApply(
                [TorchVectorizedNoise(noise_samples=noise_samples)], p=0.5
            ),
            # normalize,
        ]
    )
    return transform


def knn_accuracy(embedding, labels, n_neighbors=15):
    """
    Calculate KNN classification accuracy.

    Parameters:
    - embedding: Feature vectors
    - labels: Corresponding labels
    - n_neighbors: Number of neighbors for KNN

    Returns:
    - Accuracy score of KNN classification
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        embedding, labels, test_size=.2)

    # Create and train KNN classifier
    knn = neighbors.KNeighborsClassifier(n_neighbors)
    knn_accuracy = knn.fit(X_train, y_train).score(X_test, y_test)

    # Print and return accuracy
    print(f'kNN Accuracy: {knn_accuracy:.4f}')

    return knn_accuracy


def ari_score(embedding,  true_labels, n_clusters=None):
    """
    Calculate Adjusted Rand Index (ARI) score.

    Parameters:
    - true_labels: Ground truth labels
    - predicted_labels: Predicted cluster labels

    Returns:
    - ARI score
    """
    if n_clusters is None:
        n_clusters = np.unique(true_labels).shape[0]

    gmm = mixture.GaussianMixture(n_components=n_clusters,
                                  covariance_type='diag',
                                  random_state=42)
    gmm.fit(embedding)
    labels_predicted = gmm.predict(embedding)
    ari = metrics.adjusted_rand_score(true_labels, labels_predicted)

    # Print and return ARI score
    print(f'ARI Score: {ari:.4f}')

    return ari

def compute_discriminability(X_in, labels, class1=1, class2=2):
        """
        Compute a simple discriminability measure between two classes in a PCA-reduced dataset.

        Parameters:
        X_pca : np.ndarray
            The PCA-transformed data (samples x features).
        labels : np.ndarray
            Array of class labels corresponding to rows in X_pca.
        class1 : int
            Label for the first class.
        class2 : int
            Label for the second class.

        Returns:
        float
            The discriminability measure.
        """
        # Extract data for each class
        X_class1 = X_in[labels == class1]
        X_class2 = X_in[labels == class2]

        # Calculate means per feature
        mu1 = np.mean(X_class1, axis=0)
        mu2 = np.mean(X_class2, axis=0)

        # Calculate standard deviations per feature
        std1 = np.std(X_class1, axis=0)
        std2 = np.std(X_class2, axis=0)

        # Get difference
        pooled_std = 0.5 * (std1 + std2)
        diff = mu1 - mu2
        normalized_diff = diff / pooled_std

        # Normalize
        discrim = np.linalg.norm(normalized_diff)

        return discrim