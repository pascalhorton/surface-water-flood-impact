"""
NumPy implementation of the MiniRocket transform for univariate time series.

Reference: Dempster, Schmidt & Webb (2021), "MiniRocket: A Very Fast (Almost)
Deterministic Transform for Time Series Classification", KDD 2021.

This implementation follows the published algorithm (84 fixed kernels of
length 9 with weights in {-1, 2}, dilations spread geometrically up to the
input length, biases drawn from quantiles of convolution outputs on random
training samples, PPV pooling). It deviates from the reference code in one
point: padded/unpadded outputs alternate by kernel index instead of by
feature index, which keeps the half/half split but simplifies vectorization.
"""

import itertools
import logging

import numpy as np

logger = logging.getLogger(__name__)

KERNEL_LENGTH = 9
NUM_KERNELS = 84  # C(9, 3) combinations of the three positions with weight 2


class MiniRocket:
    """
    MiniRocket feature transform for univariate series.

    Input arrays are (n_samples, n_timesteps); the output of transform() is
    (n_samples, num_features) with values in [0, 1] (PPV pooling).

    Parameters
    ----------
    num_features: int
        The (approximate) total number of features. Rounded down to a multiple
        of 84. Default 9996 (= 84 * 119) as in the reference implementation.
    max_dilations_per_kernel: int
        The maximum number of distinct dilations per kernel (default 32).
    random_state: int|None
        Seed for the sampling of training series used to set the biases.
    """

    def __init__(self, num_features=9996, max_dilations_per_kernel=32,
                 random_state=None):
        assert num_features >= NUM_KERNELS, \
            f"num_features must be >= {NUM_KERNELS}"
        self._num_features_per_kernel = num_features // NUM_KERNELS
        self._max_dilations_per_kernel = max_dilations_per_kernel
        self._rng = np.random.default_rng(random_state)
        self._weights = self._build_kernel_weights()
        self._input_length = None
        self._dilations = None
        self._num_features_per_dilation = None
        self._biases = None  # list per dilation: (NUM_KERNELS, m_d)

    @property
    def num_features(self):
        if self._num_features_per_dilation is None:
            return None
        return int(NUM_KERNELS * self._num_features_per_dilation.sum())

    def fit(self, x):
        """
        Set the dilations from the series length and the biases from quantiles
        of convolution outputs on randomly chosen training series.

        Parameters
        ----------
        x: np.array
            The training series, (n_samples, n_timesteps) with n_timesteps >= 9.
        """
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2, "x must be 2D (n_samples, n_timesteps)"
        n_samples, input_length = x.shape
        assert input_length >= KERNEL_LENGTH, \
            f"Series length must be >= {KERNEL_LENGTH}"

        self._input_length = input_length
        self._set_dilations(input_length)

        self._biases = []
        for dilation, m in zip(self._dilations,
                               self._num_features_per_dilation):
            # One random training series per kernel to draw the bias quantiles
            idxs = self._rng.integers(0, n_samples, NUM_KERNELS)
            conv = self._convolve(x[idxs], dilation)  # (84, T, 84)
            conv = conv[np.arange(NUM_KERNELS), :, np.arange(NUM_KERNELS)]
            quantiles = self._quantiles(NUM_KERNELS * m).reshape(NUM_KERNELS, m)
            biases = np.empty((NUM_KERNELS, m), dtype=np.float32)
            valid = slice(4 * dilation, input_length - 4 * dilation)
            for k in range(NUM_KERNELS):
                series = conv[k] if k % 2 == 0 else conv[k, valid]
                biases[k] = np.quantile(series, quantiles[k])
            self._biases.append(biases)

        logger.info("MiniRocket fitted: %d features (%d dilations: %s)",
                    self.num_features, len(self._dilations),
                    self._dilations.tolist())
        return self

    def transform(self, x, chunk_size=256):
        """
        Compute the PPV features.

        Parameters
        ----------
        x: np.array
            The series, (n_samples, n_timesteps) with the same length as in fit.
        chunk_size: int
            The number of samples per vectorized block (memory control).

        Returns
        -------
        np.array
            The features, (n_samples, num_features), float32 in [0, 1].
        """
        assert self._biases is not None, "fit() must be called first"
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2 and x.shape[1] == self._input_length, \
            f"x must be (n, {self._input_length})"

        n_samples = x.shape[0]
        features = np.empty((n_samples, self.num_features), dtype=np.float32)
        for start in range(0, n_samples, chunk_size):
            sl = slice(start, min(start + chunk_size, n_samples))
            features[sl] = self._transform_chunk(x[sl])
        return features

    def _transform_chunk(self, x):
        blocks = []
        for dilation, biases in zip(self._dilations, self._biases):
            conv = self._convolve(x, dilation)  # (n, T, 84)
            m = biases.shape[1]
            ppv = np.empty((x.shape[0], NUM_KERNELS, m), dtype=np.float32)
            valid = slice(4 * dilation, self._input_length - 4 * dilation)
            for parity, time_slice in ((0, slice(None)), (1, valid)):
                kernels = np.arange(parity, NUM_KERNELS, 2)
                c = conv[:, time_slice][:, :, kernels]  # (n, T', 42)
                b = biases[kernels]  # (42, m)
                ppv[:, kernels] = (
                    c[:, :, :, None] > b[None, None, :, :]
                ).mean(axis=1, dtype=np.float32)
            blocks.append(ppv.reshape(x.shape[0], -1))
        return np.concatenate(blocks, axis=1)

    def _convolve(self, x, dilation):
        """
        Dilated convolution of x (n, T) with all 84 kernels, 'same' padding.
        Returns (n, T, 84).
        """
        n_samples, t_len = x.shape
        pad = 4 * dilation
        x_pad = np.pad(x, ((0, 0), (pad, pad)))
        taps = np.empty((n_samples, t_len, KERNEL_LENGTH), dtype=np.float32)
        for j in range(KERNEL_LENGTH):
            taps[:, :, j] = x_pad[:, j * dilation:j * dilation + t_len]
        return np.tensordot(taps, self._weights, axes=([2], [1]))

    def _set_dilations(self, input_length):
        true_max = min(self._num_features_per_kernel,
                       self._max_dilations_per_kernel)
        multiplier = self._num_features_per_kernel / true_max
        max_exponent = np.log2((input_length - 1) / (KERNEL_LENGTH - 1))
        dilations, counts = np.unique(
            np.logspace(0, max_exponent, true_max, base=2).astype(np.int32),
            return_counts=True)
        num_per_dilation = (counts * multiplier).astype(np.int32)
        remainder = self._num_features_per_kernel - num_per_dilation.sum()
        i = 0
        while remainder > 0:
            num_per_dilation[i] += 1
            remainder -= 1
            i = (i + 1) % len(num_per_dilation)
        self._dilations = dilations
        self._num_features_per_dilation = num_per_dilation

    @staticmethod
    def _build_kernel_weights():
        weights = np.full((NUM_KERNELS, KERNEL_LENGTH), -1.0, dtype=np.float32)
        for k, idxs in enumerate(
                itertools.combinations(range(KERNEL_LENGTH), 3)):
            weights[k, list(idxs)] = 2.0
        return weights

    @staticmethod
    def _quantiles(n):
        # Low-discrepancy sequence: multiples of the golden ratio, mod 1
        phi = (np.sqrt(5) + 1) / 2
        return (np.arange(1, n + 1) * phi) % 1
