"""
Class to generate the data for the Transformer model.
"""
from .impact_dl_data_generator import ImpactDlDataGenerator

import numpy as np


class ImpactTxDataGenerator(ImpactDlDataGenerator):
    def __init__(self, event_props, x_static, x_precip_hf, x_precip_daily, y,
                 batch_size=32, shuffle=True, precip_daily_days_nb=30,
                 precip_hf_time_step=60, precip_hf_days_before=1, precip_hf_days_after=1,
                 tmp_dir=None, transform_static='standardize', transform_precip='normalize',
                 log_transform_precip=True, mean_static=None, std_static=None,
                 mean_precip_hf=None, std_precip_hf=None, mean_precip_daily=None,
                 std_precip_daily=None, min_static=None, max_static=None,
                 q99_precip_hf=None, q99_precip_daily=None, debug=False):
        """
        Data generator class.

        Parameters
        ----------
        event_props: np.array
            The event properties (2D; dates and coordinates).
        x_static: np.array
            The static predictor variables (0D).
        x_precip_hf: Precipitation
            The high-frequency precipitation data.
        x_precip_daily: Precipitation
            The daily precipitation data.
        y: np.array
            The target variable.
        batch_size: int
            The batch size.
        shuffle: bool
            Whether to shuffle the data or not.
        precip_daily_days_nb: int
            The number of days of daily precipitation data.
        precip_hf_time_step: int
            The time step of the high-frequency precipitation data.
        precip_hf_days_before: int
            The number of days before the event to include in the high-frequency precipitation data.
        precip_hf_days_after: int
            The number of days after the event to include in the high-frequency precipitation data.
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
        mean_precip_hf: np.array
            The mean of the high-frequency precipitation data.
        std_precip_hf: np.array
            The standard deviation of the high-frequency precipitation data.
        mean_precip_daily: np.array
            The mean of the daily precipitation data.
        std_precip_daily: np.array
            The standard deviation of the daily precipitation data.
        min_static: np.array
            The min of the static data.
        max_static: np.array
            The max of the static data.
        q99_precip_hf: np.array
            The 99th percentile of the high-frequency precipitation data.
        q99_precip_daily: np.array
            The 99th percentile of the daily precipitation data.
        debug: bool
            Whether to run in debug mode or not (print more messages).
        """
        super().__init__(event_props, x_static, y,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         tmp_dir=tmp_dir,
                         transform_static=transform_static,
                         transform_precip=transform_precip,
                         log_transform_precip=log_transform_precip,
                         mean_static=mean_static,
                         std_static=std_static,
                         min_static=min_static,
                         max_static=max_static,
                         debug=debug)
        self.precip_hf_dim_size = None
        self.precip_daily_dim_size = None
        self.precip_daily_days_nb = precip_daily_days_nb
        self.precip_hf_time_step = precip_hf_time_step
        self.precip_hf_days_before = precip_hf_days_before
        self.precip_hf_days_after = precip_hf_days_after

        self.mean_precip_hf = mean_precip_hf
        self.std_precip_hf = std_precip_hf
        self.q99_precip_hf = q99_precip_hf

        self.mean_precip_daily = mean_precip_daily
        self.std_precip_daily = std_precip_daily
        self.q99_precip_daily = q99_precip_daily

        self.X_precip_hf = x_precip_hf
        self.X_precip_daily = x_precip_daily

        self._compute_predictor_statistics()

        if transform_static == 'standardize':
            self._standardize_static_inputs()
        elif transform_static == 'normalize':
            self._normalize_static_inputs()

        if transform_precip == 'standardize':
            self._standardize_precip_inputs()
        elif transform_precip == 'normalize':
            self._normalize_precip_inputs()

        self.on_epoch_end()  # Shuffle the data

    def get_precip_hf_length(self):
        """ Get the length of the high-frequency precipitation data. """
        if self.precip_hf_dim_size is not None:
            return self.precip_hf_dim_size

        precip_hf_dim_size = 0
        if self.X_precip_hf is not None:
            precip_hf_dim_size += self.precip_hf_days_after + self.precip_hf_days_before
            precip_hf_dim_size *= int(24 * 60 / self.precip_hf_time_step)  # Time step
            precip_hf_dim_size += 1  # Because the 1st and last time steps are included.

        self.precip_hf_dim_size = precip_hf_dim_size

        return precip_hf_dim_size

    def get_precip_daily_length(self):
        """ Get the length of the daily precipitation data. """
        if self.precip_daily_dim_size is not None:
            return self.precip_daily_dim_size

        precip_daily_dim_size = 0
        if self.X_precip_daily is not None:
            precip_daily_dim_size = self.precip_daily_days_nb

        self.precip_daily_dim_size = precip_daily_dim_size

        return precip_daily_dim_size

    def _standardize_precip_inputs(self):
        if self.X_precip_hf is not None:
            self.X_precip_hf.standardize(self.mean_precip_hf,
                                         self.std_precip_hf)
        if self.X_precip_daily is not None:
            self.X_precip_daily.standardize(self.mean_precip_daily,
                                            self.std_precip_daily)

    def _normalize_precip_inputs(self):
        if self.X_precip_hf is not None:
            self.X_precip_hf.normalize(self.q99_precip_hf)
        if self.X_precip_daily is not None:
            self.X_precip_daily.normalize(self.q99_precip_daily)

    def _compute_predictor_statistics(self):
        self._compute_static_predictor_statistics()

        if self.X_precip_hf is not None:
            # Log transform the precipitation
            if self.log_transform_precip:
                print('Log-transforming high-frequency precipitation')
                self.X_precip_hf.log_transform()

            # Load or compute the precipitation statistics
            if self.transform_precip == 'standardize':
                if self.mean_precip_hf is None or self.std_precip_hf is None:
                    self.mean_precip_hf, self.std_precip_hf = (
                        self.X_precip_hf.compute_mean_and_std_per_pixel())
            elif self.transform_precip == 'normalize':
                if self.q99_precip_hf is None:
                    self.q99_precip_hf = (
                        self.X_precip_hf.compute_quantile_per_pixel(0.99))

        if self.X_precip_daily is not None:
            # Log transform the precipitation
            if self.log_transform_precip:
                print('Log-transforming daily precipitation')
                self.X_precip_daily.log_transform()

            # Load or compute the precipitation statistics
            if self.transform_precip == 'standardize':
                if self.mean_precip_daily is None or self.std_precip_daily is None:
                    self.mean_precip_daily, self.std_precip_daily = (
                        self.X_precip_daily.compute_mean_and_std_per_pixel())
            elif self.transform_precip == 'normalize':
                if self.q99_precip_daily is None:
                    self.q99_precip_daily = (
                        self.X_precip_daily.compute_quantile_per_pixel(0.99))

    def __getitem__(self, i):
        """Generate one batch of data"""
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]

        # Select the events
        y = self.y[idxs]

        x_precip_hf = None
        x_precip_daily = None
        x_static = None

        # Select the high-frequency precipitation data
        if self.X_precip_hf is not None:
            x_precip_hf = np.zeros((self.batch_size,
                                    self.get_precip_hf_length()))

            for i_b, event in enumerate(self.event_props[idxs]):
                x_precip_hf[i_b] = self._extract_precipitation_hf(event)

        # Select the daily precipitation data
        if self.X_precip_daily is not None:
            x_precip_daily = np.zeros((self.batch_size,
                                       self.get_precip_daily_length()))

            for i_b, event in enumerate(self.event_props[idxs]):
                x_precip_daily[i_b] = self._extract_precipitation_daily(event)

        # Select the static data
        if self.X_static is not None:
            x_static = self.X_static[idxs, :]

        if self.X_static is None:
            if self.X_precip_hf is None:
                return x_precip_daily, y
            elif self.X_precip_daily is None:
                return x_precip_hf, y
            else:
                return (x_precip_daily, x_precip_hf), y

        if self.X_precip_hf is None:
            if self.X_precip_daily is None:
                return x_static, y
            elif self.X_static is None:
                return x_precip_daily, y
            else:
                return (x_precip_daily, x_static), y

        if self.X_precip_daily is None:
            if self.X_precip_hf is None:
                return x_static, y
            elif self.X_static is None:
                return x_precip_hf, y
            else:
                return (x_static, x_precip_hf), y

        return (x_precip_daily, x_precip_hf, x_static), y

    def _extract_precipitation_hf(self, event):
        # Temporal selection
        t_start = event[0] - np.timedelta64(self.precip_hf_days_before, 'D')
        t_end = event[0] + np.timedelta64(self.precip_hf_days_after, 'D')

        # Spatial domain
        x = event[1]
        y = event[2]

        # Select the corresponding precipitation data
        cid = event[3]
        x_precip_ev = self.X_precip_hf.get_data_chunk(
            t_start, t_end, x, x, y, y, cid
        )

        # Remove axis with length 1
        x_precip_ev = np.squeeze(x_precip_ev)

        # Handle missing precipitation data
        if x_precip_ev.shape[0] != self.get_precip_hf_length():
            self.warning_counter += 1
            self._analyze_precip_shape_difference(
                event, x_precip_ev, x_precip_ev.shape[0],
                self.get_precip_hf_length())

            diff = x_precip_ev.shape[0] - self.get_precip_hf_length()
            if abs(diff / self.get_precip_hf_length()) > 0.1:  # 10% tolerance
                if self.debug:
                    print(f"Warning: too many missing timesteps ({diff}).")

                x_precip_ev = self._create_empty_precip_block(
                    self.get_precip_hf_length())

            else:
                empty_block = self._create_empty_precip_block(-diff)
                x_precip_ev = np.concatenate([x_precip_ev, empty_block], axis=-1)

        return x_precip_ev

    def _extract_precipitation_daily(self, event):
        # Temporal selection
        t_shift = np.timedelta64(self.precip_hf_days_before + 1, 'D')
        t_days_nb = np.timedelta64(self.precip_daily_days_nb - 1, 'D')
        t_start = event[0] - t_shift - t_days_nb
        t_end = event[0] - t_shift

        # Spatial domain
        x = event[1]
        y = event[2]

        # Select the corresponding precipitation data
        cid = event[3]
        x_precip_ev = self.X_precip_daily.get_data_chunk(
            t_start, t_end, x, x, y, y, cid
        )

        # Remove axis with length 1
        x_precip_ev = np.squeeze(x_precip_ev)

        # Handle missing precipitation data
        if x_precip_ev.shape[0] != self.get_precip_daily_length():
            self.warning_counter += 1
            self._analyze_precip_shape_difference(
                event, x_precip_ev, x_precip_ev.shape[0],
                self.get_precip_daily_length())

            diff = x_precip_ev.shape[0] - self.get_precip_daily_length()
            if abs(diff / self.get_precip_daily_length()) > 0.1:
                if self.debug:
                    print(f"Warning: too many missing timesteps ({diff}).")

                x_precip_ev = self._create_empty_precip_block(
                    self.get_precip_daily_length())

            else:
                empty_block = self._create_empty_precip_block(-diff)
                x_precip_ev = np.concatenate([x_precip_ev, empty_block], axis=-1)

        return x_precip_ev
