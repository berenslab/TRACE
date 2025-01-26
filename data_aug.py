import math
import numpy as np

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

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

