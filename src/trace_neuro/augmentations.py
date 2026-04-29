import math
import numpy as np
import torch
from torchvision import transforms

class AmpJitter(object):
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

    def __call__(self, sample):
        amp_jit_value = np.random.uniform(self.lo, self.hi)
        #amp_jit = np.array([amp_jit_value for i in range(sample.shape[0])])
        sample_transformed = (sample * amp_jit_value).astype('float32')

        return sample_transformed

class TempJitter(object):
    """
    Temporal jittering of the sample by shifting the time axis.

    Args:
        shift_n_bins (int): number of bins to shift the sample

    Returns:
        sample_transformed (np.array): sample with shifted time axis
    """
    def __init__(self, shift_n_bins = 3):
        self.shift_n_bins = shift_n_bins

    def __call__(self, sample):
        # Generate the shift
        shift_ = (2 * np.random.binomial(1, 0.5) - 1) * np.random.uniform(0, self.shift_n_bins)
        # Calculate integer shift value
        int_shift = int(math.ceil(shift_)) if shift_ >= 0 else -int(math.floor(shift_))
        # Apply the shift without padding first
        if shift_ >= 0:
            # Positive shift: roll forward
            #sample_transformed = np.roll(sample, int_shift)
            sample_transformed = np.concatenate([sample[-int_shift:], sample[:-int_shift]])
        else:
            # Negative shift: roll backward and take values from the beginning to the end
            # todo: Do we really need this separate case? Or could we omit the int_shift step?
            sample_transformed = np.concatenate([sample[int_shift:], sample[:int_shift]])

        return sample_transformed.astype('float32')

class Noise(object):
    """
    Add Gaussian noise to the sample based on a temporal covariance matrix. Noise samples
    have been pre-computed np.random.multivariate_normal(mean=np.zeros(cov_matrix.shape[0]), cov=cov_matrix)

    Args:
        noise_scale (float): scale of the noise

    Returns:
        sample_transformed (np.array): sample with added noise
    """
    def __init__(self, noise_scale=.5, noise_samples=None):
        self.noise_scale = noise_scale
        self.noise_samples = noise_samples

    def __call__(self, sample):
        # Generate Gaussian noise based on the temporal covariance matrix
        #noise = np.random.multivariate_normal(mean=np.zeros(self.cov_matrix.shape[0]),
        #                                      cov=self.cov_matrix)
        # Randomly select one of the pre-computed noise samples
        noise = self.noise_samples[np.random.randint(0, self.noise_samples.shape[0]), :]
        # Scale the noise and add to the original sample
        sample_transformed = sample + self.noise_scale * noise

        return sample_transformed.astype('float32')


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