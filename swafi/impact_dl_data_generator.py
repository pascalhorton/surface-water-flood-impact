"""
Class to generate data for the deep learning models.
"""

import keras
import numpy as np
from pathlib import Path


class ImpactDlDataGenerator(keras.utils.Sequence):
    def __init__(self, event_props, x_static, y, batch_size=32, shuffle=True,
                 tmp_dir=None, transform_static='standardize',
                 transform_precip='normalize', log_transform_precip=True,
                 mean_static=None, std_static=None, min_static=None,
                 max_static=None, debug=False):
        """
        Data generator class.
        Template from:
        https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Adapted by :
        https://github.com/pangeo-data/WeatherBench/blob/master/src/train_nn.py

        Parameters
        ----------
        event_props: np.array
            The event properties (2D; dates and coordinates).
        x_static: np.array
            The static predictor variables (0D).
        y: np.array
            The target variable.
        batch_size: int
            The batch size.
        shuffle: bool
            Whether to shuffle the data or not.
        tmp_dir: Path
            The temporary directory to use.
        transform_static: str
            The transformation to apply to the static data.
            Options: 'normalize' or 'standardize'.
        transform_precip: str
            The transformation to apply to the 3D data.
            Options: 'normalize' or 'standardize'.
        log_transform_precip: bool
            Whether to log-transform the precipitation data or not.
        mean_static: np.array
            The mean of the static data.
        std_static: np.array
            The standard deviation of the static data.
        min_static: np.array
            The min of the static data.
        max_static: np.array
            The max of the static data.
        debug: bool
            Whether to run in debug mode or not (print more messages).
        """
        super().__init__()
        self.tmp_dir = tmp_dir
        self.event_props = event_props
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.debug = debug

        self.transform_static = transform_static
        self.transform_precip = transform_precip
        self.log_transform_precip = log_transform_precip

        self.mean_static = mean_static
        self.std_static = std_static
        self.min_static = min_static
        self.max_static = max_static

        self.X_static = x_static

        self.n_samples = self.y.shape[0]
        self.idxs = np.arange(self.n_samples)

    def reduce_negatives(self, factor):
        """
        Reduce the number of negative events. It is done by randomly subsampling
        indices of negative events, but does not remove data.

        Parameters
        ----------
        factor: int
            The factor by which to reduce the number of negative events.
        """
        if factor == 1:
            return

        # Select the indices of the negative events
        idxs_neg = np.where(self.y == 0)[0]
        n_neg = idxs_neg.shape[0]
        n_neg_new = int(n_neg / factor)
        idxs_neg_new = np.random.choice(idxs_neg, size=n_neg_new, replace=False)

        # Select the indices of the positive events
        idxs_pos = np.where(self.y > 0)[0]

        # Concatenate the indices
        self.idxs = np.concatenate([idxs_neg_new, idxs_pos])
        self.n_samples = self.idxs.shape[0]

        print(f"Reduced the number of negative events from {n_neg} to {n_neg_new}")
        print(f"Number of positive events: {idxs_pos.shape[0]}")

        # Shuffle
        np.random.shuffle(self.idxs)

    def get_number_of_batches_for_full_dataset(self):
        """
        Get the number of batches for the full data (i.e., without shuffling or
        negative event removal).

        Returns
        -------
        The number of batches.
        """

        return int(np.floor(len(self.y) / self.batch_size))

    def get_ordered_batch_from_full_dataset(self, i):
        """
        Get a batch of data from the full data (i.e., without shuffling or negative
        event removal).

        Parameters
        ----------
        i : int
            The batch index.

        Returns
        -------
        The batch of data.
        """

        # Save the original indices
        idxs_orig = self.idxs

        # Reset the indices
        self.idxs = np.arange(len(self.y))

        batch = self.__getitem__(i)

        # Restore the original indices
        self.idxs = idxs_orig

        return batch

    def _standardize_static_inputs(self):
        if self.X_static is not None:
            self.X_static = (self.X_static - self.mean_static) / self.std_static

    def _normalize_static_inputs(self):
        if self.X_static is not None:
            self.X_static = ((self.X_static - self.min_static) /
                             (self.max_static - self.min_static))

    def _compute_static_predictor_statistics(self):
        if self.X_static is not None:
            print('Computing/assigning static predictor statistics')
            if self.transform_static == 'standardize':
                # Compute the mean and standard deviation of the static data
                if self.mean_static is None:
                    self.mean_static = np.mean(self.X_static, axis=0)
                if self.std_static is None:
                    self.std_static = np.std(self.X_static, axis=0)
            elif self.transform_static == 'normalize':
                # Compute the min and max of the static data
                if self.min_static is None:
                    self.min_static = np.min(self.X_static, axis=0)
                if self.max_static is None:
                    self.max_static = np.max(self.X_static, axis=0)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.n_samples / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch and reset the warning counter."""
        self.warning_counter = 0
        if self.shuffle:
            np.random.shuffle(self.idxs)
