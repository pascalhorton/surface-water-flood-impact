"""
Class to generate data for the deep learning models.
"""

import logging

import keras
import numpy as np
from pathlib import Path


logger = logging.getLogger(__name__)


class ImpactDlDataGenerator(keras.utils.Sequence):
    def __init__(self, event_props, x_static, y, batch_size=32, shuffle=True,
                 tmp_dir=None, transform_static='standardize',
                 transform_precip='normalize', log_transform_precip=True,
                 mean_static=None, std_static=None, min_static=None,
                 max_static=None, batch_pos_ratio=None, debug=False):
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
        self.warning_counter = 0
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

        self.n_samples = self.X_static.shape[0]
        self.idxs = np.arange(self.n_samples)

        # Epoch-reshuffled negative subsampling (factor_neg_reduction > 1)
        self._factor_neg_reduction = 1
        self._all_idxs_neg = None

        # Stratified batch sampling: guarantee positives in every batch
        self.batch_pos_ratio = batch_pos_ratio
        self._idxs_pos = None
        self._idxs_neg = None
        if batch_pos_ratio is not None:
            self._idxs_pos = np.where(self.y > 0)[0]
            self._idxs_neg = np.where(self.y == 0)[0]
            np.random.shuffle(self._idxs_neg)
            n_pos_per_batch = max(1, int(self.batch_size * batch_pos_ratio))
            n_neg_per_batch = self.batch_size - n_pos_per_batch
            n_batches = max(1, len(self._idxs_neg) // n_neg_per_batch)
            oversampling = (n_pos_per_batch * n_batches) / max(1, len(self._idxs_pos))
            logger.info(
                "Stratified batching: pos_ratio=%.3f, %d pos, %d neg, "
                "%d pos/batch, ~%d batches/epoch, oversampling=%.1f×",
                batch_pos_ratio, len(self._idxs_pos), len(self._idxs_neg),
                n_pos_per_batch, n_batches, oversampling)
            if oversampling > 5:
                logger.warning(
                    "Positive oversampling factor %.1f× is high (target: 1–3×). "
                    "Consider lowering --batch-pos-ratio to avoid memorization.",
                    oversampling)

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

        if self.batch_pos_ratio is not None:
            logger.warning(
                "reduce_negatives(factor=%d) has no effect when batch_pos_ratio is set: "
                "the stratified generator uses self._idxs_neg (all negatives), not self.idxs. "
                "Use batch_pos_ratio alone to control training speed and class balance.",
                factor)
            return

        self._factor_neg_reduction = factor
        self._all_idxs_neg = np.where(self.y == 0)[0]
        n_neg = len(self._all_idxs_neg)
        n_neg_per_epoch = int(n_neg / factor)
        logger.info(
            "Negative subsampling enabled: factor=%d, %d -> %d negatives per epoch "
            "(subset reshuffled each epoch; all negatives seen over ~%d epochs)",
            factor, n_neg, n_neg_per_epoch, factor)
        self._resample_negatives()

    def _resample_negatives(self):
        """Draw a fresh random subset of negatives. Called at init and each epoch end."""
        n_neg_new = int(len(self._all_idxs_neg) / self._factor_neg_reduction)
        idxs_neg_new = np.random.choice(
            self._all_idxs_neg, size=n_neg_new, replace=False)
        idxs_pos = np.where(self.y > 0)[0]
        self.idxs = np.concatenate([idxs_neg_new, idxs_pos])
        self.n_samples = self.idxs.shape[0]
        np.random.shuffle(self.idxs)

    def get_number_of_batches_for_full_dataset(self):
        """
        Get the number of batches for the full data (i.e., without shuffling or
        negative event removal).

        Returns
        -------
        The number of batches.
        """

        return int(np.ceil(len(self.y) / self.batch_size))

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
        idxs_full = np.arange(len(self.y))
        i_start = i * self.batch_size
        i_end = min((i + 1) * self.batch_size, len(self.y) - 1)
        idxs = idxs_full[i_start:i_end]

        return self._generate_batch(idxs)

    def get_batch_for_cid(self, cid):
        """
        Get a batch of data from the full data (i.e., without shuffling or negative
        event removal) for a given cid.

        Parameters
        ----------
        cid : int
            The cell id.

        Returns
        -------
        The batch of data.
        """
        idxs = np.where(self.event_props[:, 3] == cid)[0]

        return self._generate_batch(idxs)

    def get_batch_for_indices(self, idxs):
        """
        Get a batch of data from the full data (i.e., without shuffling or negative
        event removal) for the given event indices.

        Parameters
        ----------
        idxs : np.ndarray
            The event indices into the full data.

        Returns
        -------
        The batch of data.
        """
        return self._generate_batch(np.asarray(idxs))

    def get_event_dates_for_cid(self, cid):
        """
        Get all event dates for a given cid.

        Parameters
        ----------
        cid : int
            The cell id.

        Returns
        -------
        The event dates.
        """
        idxs = np.where(self.event_props[:, 3] == cid)[0]

        return self.event_props[idxs, 0]

    def _standardize_static_inputs(self):
        if self.X_static is not None:
            self.X_static = (self.X_static - self.mean_static) / self.std_static

    def _normalize_static_inputs(self):
        if self.X_static is not None:
            self.X_static = ((self.X_static - self.min_static) /
                             (self.max_static - self.min_static))

    def _compute_static_predictor_statistics(self):
        if self.X_static is not None:
            logger.info('Computing/assigning static predictor statistics')
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

    def _create_empty_precip_block(self, shape):
        """Create an empty precipitation block. Log-transform if needed."""
        empty_block = np.zeros(shape)
        if self.log_transform_precip:
            empty_block = (np.log1p(empty_block)).astype('float32')

        return empty_block

    def _analyze_precip_shape_difference(self, event, precip_ev, data_length,
                                         expected_length):
        """Analyze the precipitation data shape difference."""
        if data_length > expected_length:
            logger.error("Data array larger than expected: %s > %s", data_length, expected_length)
            logger.error("Event: %s", event)
            logger.error("Data shape: %s", precip_ev.shape)
            logger.error("Data: %s", precip_ev)
            raise ValueError("Data array larger than expected.")

        if self.debug:
            logger.debug("Shape mismatch: expected: %s != got: %s", expected_length, data_length)
            logger.debug("Event: %s", event)

        if self.warning_counter in [10, 50, 100, 500, 1000]:
            logger.warning("Shape mismatch: expected: %s != got: %s", expected_length, precip_ev.shape[-1])
            logger.warning("%s events with shape mismatch (e.g., missing precipitation data).",
                           self.warning_counter)

        if self.warning_counter > 1000:
            raise ValueError("Too many issues with precipitation data.")

    def __len__(self):
        """Denotes the number of batches per epoch."""
        if self.batch_pos_ratio is not None:
            n_pos_per_batch = max(1, int(self.batch_size * self.batch_pos_ratio))
            n_neg_per_batch = self.batch_size - n_pos_per_batch
            return max(1, len(self._idxs_neg) // n_neg_per_batch)
        return int(np.floor(self.n_samples / self.batch_size))

    def _get_batch_idxs(self, i):
        """Return sample indices for batch i.

        In stratified mode: draws positives with replacement and slices through
        all negatives sequentially, guaranteeing at least one positive per batch.
        In standard mode: sequential slice of the (optionally shuffled) index array.
        """
        if self.batch_pos_ratio is None:
            return self.idxs[i * self.batch_size:(i + 1) * self.batch_size]

        n_pos_per_batch = max(1, int(self.batch_size * self.batch_pos_ratio))
        n_neg_per_batch = self.batch_size - n_pos_per_batch

        pos_idxs = np.random.choice(self._idxs_pos, size=n_pos_per_batch, replace=True)
        start = i * n_neg_per_batch
        neg_idxs = self._idxs_neg[start:start + n_neg_per_batch]

        combined = np.concatenate([pos_idxs, neg_idxs])
        np.random.shuffle(combined)
        return combined

    def on_epoch_end(self):
        """Updates indexes after each epoch and resets the warning counter."""
        self.warning_counter = 0
        if self.batch_pos_ratio is not None:
            np.random.shuffle(self._idxs_neg)
        elif self._factor_neg_reduction > 1:
            self._resample_negatives()
        elif self.shuffle:
            np.random.shuffle(self.idxs)

    def __getitem__(self, index):
        raise NotImplementedError("This method should be implemented in subclasses.")
