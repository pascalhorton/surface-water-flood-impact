"""
Class to compute the impact function.
"""

import hashlib
import pickle
import keras
import os
import numpy as np
from pathlib import Path


class DataGenerator(keras.utils.Sequence):
    def __init__(self, event_props, x_static, x_precip, x_dem, y, batch_size=32,
                 shuffle=True, use_pickle_events_precip_data=False,
                 use_pickle_full_precip_data=True, precip_window_size=2,
                 precip_resolution=1, precip_time_step=12, precip_days_before=1,
                 precip_days_after=1, tmp_dir=None, transform_static='standardize',
                 transform_2d='normalize', precip_transformation_domain='per-pixel',
                 log_transform_precip=True, mean_static=None, std_static=None,
                 mean_precip=None, std_precip=None, min_static=None,
                 max_static=None, q95_precip=None, debug=False):
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
        x_precip: xarray.Dataset
            The precipitation fields (3D).
        x_dem: xarray.DataArray
            The DEM (2D).
        y: np.array
            The target variable.
        batch_size: int
            The batch size.
        shuffle: bool
            Whether to shuffle the data or not.
        use_pickle_events_precip_data: bool
            Whether to use pickle files to handle the precipitation data for each event.
            This option is not recommended as it generates a lot of files and is not
            significantly faster.
        use_pickle_full_precip_data: bool
            Whether to load the full data into memory and dump it to a pickle file.
            This option is recommended as it speeds up the data loading.
        precip_window_size: int
            The window size for the 2D predictors [km].
        precip_resolution: int
            The desired grid resolution of the precipitation data [km].
        precip_time_step: int
            The desired time step of the precipitation data [h].
        precip_days_before: int
            The number of days before the event to include in the 2D predictors.
        precip_days_after: int
            The number of days after the event to include in the 2D predictors.
        tmp_dir: Path
            The temporary directory to use.
        transform_static: str
            The transformation to apply to the static data.
            Options: 'normalize' or 'standardize'.
        transform_2d: str
            The transformation to apply to the 2D data.
            Options: 'normalize' or 'standardize'.
        precip_transformation_domain: str
            How to apply the transformation of the precipitation data:
            'domain-average', or 'per-pixel'.
        log_transform_precip: bool
            Whether to log-transform the precipitation data or not.
        mean_static: np.array
            The mean of the static data.
        std_static: np.array
            The standard deviation of the static data.
        mean_precip: np.array
            The mean of the precipitation data.
        std_precip: np.array
            The standard deviation of the precipitation data.
        min_static: np.array
            The min of the static data.
        max_static: np.array
            The max of the static data.
        q95_precip: np.array
            The 95th percentile of the precipitation data.
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
        self.warning_counter = 0
        self.channels_nb = None
        self.use_pickle_events_precip_data = use_pickle_events_precip_data
        self.use_pickle_full_precip_data = use_pickle_full_precip_data
        self.precip_window_size = precip_window_size
        self.precip_resolution = precip_resolution
        self.precip_time_step = precip_time_step
        self.precip_days_before = precip_days_before
        self.precip_days_after = precip_days_after

        self.transform_static = transform_static
        self.transform_2d = transform_2d
        self.precip_transformation_domain = precip_transformation_domain
        self.log_transform_precip = log_transform_precip

        self.mean_static = mean_static
        self.std_static = std_static
        self.mean_precip = mean_precip
        self.std_precip = std_precip
        self.min_static = min_static
        self.max_static = max_static
        self.q95_precip = q95_precip

        self.mean_dem = None
        self.std_dem = None
        self.min_dem = None
        self.max_dem = None
        self.q25_dem = None
        self.q50_dem = None
        self.q75_dem = None

        self.X_static = x_static
        self.X_precip = x_precip
        self.X_dem = x_dem

        if self.X_precip is not None:
            self._restrict_spatial_domain()
            self._restrict_temporal_selection()

        self._compute_predictor_statistics()

        if transform_static == 'standardize':
            self._standardize_static_inputs()
        elif transform_static == 'normalize':
            self._normalize_static_inputs()

        if transform_2d == 'standardize':
            self._standardize_2d_inputs()
        elif transform_2d == 'normalize':
            self._normalize_2d_inputs()

        self.n_samples = self.y.shape[0]
        self.idxs = np.arange(self.n_samples)

        self.on_epoch_end()

        if self.X_precip is None:
            return

        if self.X_dem is not None:
            self.X_dem.load()

    def prepare_precip_data(self):
        if self.use_pickle_events_precip_data:
            print('Pre-loading precipitation events')
            # Check if all the pickle files exist.
            full_precip_data_loaded = False
            for i, event in enumerate(self.event_props):
                file_precip = self._get_precip_event_filename(event)
                if not file_precip.exists():
                    if not full_precip_data_loaded:
                        print('Loading data into RAM')
                        if self.use_pickle_full_precip_data:
                            self._load_dump_precip_data()
                        full_precip_data_loaded = True

                    # Create directories if it does not exist
                    dir_name = os.path.dirname(file_precip)
                    os.makedirs(dir_name, exist_ok=True)

                    x_2d = self._extract_precipitation(event)
                    with open(file_precip, 'wb') as f:
                        pickle.dump(x_2d, f)

        elif self.use_pickle_full_precip_data:
            print('Loading data into RAM')
            self._load_dump_precip_data()

    def get_channels_nb(self):
        """ Get the number of channels of the 2D predictors. """
        if self.channels_nb is not None:
            return self.channels_nb

        input_2d_channels = 0
        if self.X_precip is not None:
            input_2d_channels += self.precip_days_after + self.precip_days_before
            input_2d_channels *= int(24 / self.precip_time_step)  # Time step
            input_2d_channels += 1  # Because the 1st and last time steps are included.
        if self.X_dem is not None:
            input_2d_channels += 1  # Add one for the DEM layer

        self.channels_nb = input_2d_channels

        return input_2d_channels

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

    def _standardize_2d_inputs(self):
        if self.X_precip is not None:
            self.X_precip['precip'] = ((self.X_precip['precip'] - self.mean_precip) /
                                       self.std_precip)
        if self.X_dem is not None:
            self.X_dem = (self.X_dem - self.mean_dem) / self.std_dem

    def _normalize_static_inputs(self):
        if self.X_static is not None:
            self.X_static = ((self.X_static - self.min_static) /
                             (self.max_static - self.min_static))

    def _normalize_2d_inputs(self):
        if self.X_precip is not None:
            self.X_precip['precip'] = self.X_precip['precip'] / self.q95_precip
        if self.X_dem is not None:
            self.X_dem = (self.X_dem - self.min_dem) / (self.max_dem - self.min_dem)

    def _load_dump_precip_data(self):
        """ Load all the precipitation data into memory. """
        if self.tmp_dir is None:
            raise ValueError('tmp_dir must be specified')

        file_hash = self._compute_hash_precip_full()
        file_precip = self.tmp_dir / f'precip_{file_hash}.pickle'

        # If pickle file exists, load it
        if file_precip.exists():
            print('Loading precipitation data from pickle file')
            with open(file_precip, 'rb') as f:
                self.X_precip = pickle.load(f)
            return

        # Otherwise, load the data and save it to a pickle file
        print('Loading precipitation data from netCDF files')
        self.X_precip.load()
        with open(file_precip, 'wb') as f:
            pickle.dump(self.X_precip, f)

    def _restrict_spatial_domain(self):
        """ Restrict the spatial domain of the precipitation and DEM data. """
        precip_window_size_m = self.precip_window_size * 1000
        x_min = self.event_props[:, 1].min() - precip_window_size_m / 2
        x_max = self.event_props[:, 1].max() + precip_window_size_m / 2
        y_min = self.event_props[:, 2].min() - precip_window_size_m / 2
        y_max = self.event_props[:, 2].max() + precip_window_size_m / 2
        if self.X_precip is not None:
            self.X_precip = self.X_precip.sel(
                x=slice(x_min, x_max),
                y=slice(y_max, y_min)
            )
        if self.X_dem is not None:
            self.X_dem = self.X_dem.sel(
                x=slice(x_min, x_max),
                y=slice(y_max, y_min)
            )

    def _restrict_temporal_selection(self):
        """ Restrict the temporal selection of the precipitation data. """
        if self.X_precip is not None:
            # Select the min/max dates
            t_min = self.event_props[:, 0].min() - np.timedelta64(
                self.precip_days_before, 'D')
            t_max = self.event_props[:, 0].max() + np.timedelta64(
                self.precip_days_after, 'D')

            # Round to the year start/end
            t_min = np.datetime64(f'{t_min.year}-01-01')
            t_max = np.datetime64(f'{t_max.year}-12-31')

            # Select the corresponding precipitation data
            self.X_precip = self.X_precip.sel(
                time=slice(t_min, t_max)
            )

    def _compute_predictor_statistics(self):
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

        if self.X_dem is not None:
            print('Computing DEM predictor statistics')
            if self.transform_2d == 'standardize':
                # Compute the mean and standard deviation of the DEM (non-temporal)
                self.mean_dem = self.X_dem.mean(('x', 'y')).compute().values
                self.std_dem = self.X_dem.std(('x', 'y')).compute().values
            elif self.transform_2d == 'normalize':
                # Compute the min and max of the DEM (non-temporal)
                self.min_dem = self.X_dem.min(('x', 'y')).compute().values
                self.max_dem = self.X_dem.max(('x', 'y')).compute().values

        if self.X_precip is None:
            return

        # Log transform the precipitation
        if self.log_transform_precip:
            print('Log-transforming precipitation')
            self.X_precip['precip'] = np.log(self.X_precip['precip'] + 1e-10)

        # Compute the mean and standard deviation of the precipitation
        if (self.transform_2d == 'standardize' and self.mean_precip is not None
                and self.std_precip is not None):
            return

        if self.transform_2d == 'normalize' and self.q95_precip is not None:
            return

        # If pickle file exists, load it
        file_hash = self._compute_hash_precip_full()
        file_mean_precip = self.tmp_dir / f'mean_precip_{file_hash}.pickle'
        file_std_precip = self.tmp_dir / f'std_precip_{file_hash}.pickle'
        file_q95_precip = self.tmp_dir / f'q95_precip_{file_hash}.pickle'

        if (self.transform_2d == 'standardize' and file_mean_precip.exists()
                and file_std_precip.exists()):
            print('Loading precipitation statistics from pickle files')
            with open(file_mean_precip, 'rb') as f:
                self.mean_precip = pickle.load(f)
            with open(file_std_precip, 'rb') as f:
                self.std_precip = pickle.load(f)
            return

        if self.transform_2d == 'normalize' and file_q95_precip.exists():
            print('Loading precipitation statistics from pickle files')
            with open(file_q95_precip, 'rb') as f:
                self.q95_precip = pickle.load(f)
            return

        # Otherwise, compute the statistics and save them to a pickle file
        print('Computing precipitation statistics')
        if self.precip_transformation_domain == 'domain-average':
            if self.transform_2d == 'standardize':
                self.mean_precip = self.X_precip['precip'].mean(
                    ('time', 'x', 'y')).compute().values
                self.std_precip = self.X_precip['precip'].std('time').mean(
                    ('x', 'y')).compute().values
            elif self.transform_2d == 'normalize':
                self.q95_precip = self.X_precip['precip'].quantile(
                    0.95, dim=('time', 'x', 'y')).compute().values
        elif self.precip_transformation_domain == 'per-pixel':
            if self.transform_2d == 'standardize':
                self.mean_precip = self.X_precip['precip'].mean(
                    'time').compute().values
                self.std_precip = self.X_precip['precip'].std(
                    'time').compute().values
            elif self.transform_2d == 'normalize':
                self.q95_precip = self.X_precip['precip'].chunk(dict(time=-1)).quantile(
                    0.95, dim='time').compute().values
        else:
            raise ValueError(
                f'Unknown option: {self.precip_transformation_domain}')

        # Save to pickle file
        if self.transform_2d == 'standardize':
            with open(file_mean_precip, 'wb') as f:
                pickle.dump(self.mean_precip, f)
            with open(file_std_precip, 'wb') as f:
                pickle.dump(self.std_precip, f)
        elif self.transform_2d == 'normalize':
            with open(file_q95_precip, 'wb') as f:
                pickle.dump(self.q95_precip, f)

    def _compute_hash_precip_full(self):
        tag_data = (
                pickle.dumps(self.transform_2d) +
                pickle.dumps(self.precip_transformation_domain) +
                pickle.dumps(self.log_transform_precip) +
                pickle.dumps(self.X_precip['x']) +
                pickle.dumps(self.X_precip['y']) +
                pickle.dumps(self.X_precip['time'][0]) +
                pickle.dumps(self.X_precip['time'][-1]) +
                pickle.dumps(self.X_precip['precip'].shape))

        return hashlib.md5(tag_data).hexdigest()

    def _compute_hash_precip_event(self, event):
        with_dem = self.X_dem is not None

        tag_data = (
                pickle.dumps(event) +
                pickle.dumps(self.precip_window_size) +
                pickle.dumps(self.precip_resolution) +
                pickle.dumps(self.precip_time_step) +
                pickle.dumps(self.precip_days_before) +
                pickle.dumps(self.precip_days_after) +
                pickle.dumps(self.get_channels_nb()) +
                pickle.dumps(self.transform_2d) +
                pickle.dumps(self.precip_transformation_domain) +
                pickle.dumps(self.log_transform_precip) +
                pickle.dumps(with_dem) +
                pickle.dumps(self.X_precip['x']) +
                pickle.dumps(self.X_precip['y']))

        return hashlib.md5(tag_data).hexdigest()

    def _get_precip_event_filename(self, event):
        file_hash = self._compute_hash_precip_event(event)
        return (self.tmp_dir / 'precip_events' / f'{file_hash[0:2]}' /
                f'{file_hash}.pickle')

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        """Generate one batch of data"""
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]

        # Select the events
        y = self.y[idxs]

        x_2d = None
        x_static = None

        # Select the 2D data
        if self.X_precip is not None or self.use_pickle_events_precip_data:
            pixels_nb = int(self.precip_window_size / self.precip_resolution)
            x_2d = np.zeros((self.batch_size,
                             pixels_nb,
                             pixels_nb,
                             self.get_channels_nb()))

            if self.use_pickle_events_precip_data:
                for i_b, event in enumerate(self.event_props[idxs]):
                    file_precip = self._get_precip_event_filename(event)
                    with open(file_precip, 'rb') as f:
                        x_2d[i_b] = pickle.load(f)

            else:
                for i_b, event in enumerate(self.event_props[idxs]):
                    x_2d[i_b] = self._extract_precipitation(event)

            if self.X_static is None:
                return x_2d, y

        # Select the static data
        if self.X_static is not None:
            x_static = self.X_static[idxs, :]

            if self.X_precip is None:
                return x_static, y

        return (x_2d, x_static), y

    def _extract_precipitation(self, event):
        precip_window_size_m = self.precip_window_size * 1000
        pixels_nb = int(self.precip_window_size / self.precip_resolution)

        # Temporal selection
        t_start = event[0] - np.timedelta64(self.precip_days_before, 'D')
        t_end = event[0] + np.timedelta64(self.precip_days_after, 'D')

        # Spatial domain
        x_start = event[1] - precip_window_size_m / 2
        x_end = event[1] + precip_window_size_m / 2
        y_start = event[2] + precip_window_size_m / 2
        y_end = event[2] - precip_window_size_m / 2

        # Select the corresponding precipitation data (5 days prior the event)
        x_precip_ev = self.X_precip['precip'].sel(
            time=slice(t_start, t_end),
            x=slice(x_start, x_end),
            y=slice(y_start, y_end)
        ).to_numpy()

        # Move the time axis to the last position
        x_precip_ev = np.moveaxis(x_precip_ev, 0, -1)

        if self.X_dem is not None:
            # Select the corresponding DEM data in the window
            x_dem_ev = self.X_dem.sel(
                x=slice(x_start, x_end),
                y=slice(y_start, y_end)
            ).to_numpy()

            # Replace the NaNs by zeros
            x_dem_ev = np.nan_to_num(x_dem_ev)

            # Add a new axis for the channels
            x_dem_ev = np.expand_dims(x_dem_ev, axis=-1)

            # Concatenate
            x_2d_ev = np.concatenate([x_dem_ev, x_precip_ev], axis=-1)
        else:
            x_2d_ev = x_precip_ev

        # If too large, remove the last line(s) or column(s)
        if x_2d_ev.shape[0] > pixels_nb:
            x_2d_ev = x_2d_ev[:pixels_nb, :, :]
        if x_2d_ev.shape[1] > pixels_nb:
            x_2d_ev = x_2d_ev[:, :pixels_nb, :]

        # Handle missing precipitation data
        if x_2d_ev.shape[2] != self.get_channels_nb():
            self.warning_counter += 1

            if self.debug:
                print(f"Shape mismatch: actual: {x_2d_ev.shape[2]} !="
                      f" expected: {self.get_channels_nb()}")
                print(f"Event: {event}")

            if self.warning_counter in [10, 50, 100, 500, 1000, 5000, 10000]:
                print(f"Shape mismatch: actual: {x_2d_ev.shape[2]} !="
                      f" expected: {self.get_channels_nb()}")
                print(f"Warning: {self.warning_counter} events with "
                      f"shape missmatch (e.g., missing precipitation data).")

            diff = x_2d_ev.shape[2] - self.get_channels_nb()
            if abs(diff / self.get_channels_nb()) > 0.1:  # 10% tolerance
                if self.debug:
                    print(f"Warning: too many missing channels ({diff}).")

                pixels_nb = int(self.precip_window_size / self.precip_resolution)
                x_2d_ev = np.zeros((pixels_nb, pixels_nb, self.get_channels_nb()))

            else:
                if x_2d_ev.shape[2] > self.get_channels_nb():  # Loaded too many
                    x_2d_ev = x_2d_ev[:, :, :-diff]

                elif x_2d_ev.shape[2] < self.get_channels_nb():  # Loaded too few
                    diff = -diff
                    x_2d_ev = np.concatenate(
                        [x_2d_ev, np.zeros((x_2d_ev.shape[0],
                                            x_2d_ev.shape[1],
                                            diff))], axis=-1)

        return x_2d_ev

    def on_epoch_end(self):
        """Updates indexes after each epoch and reset the warning counter."""
        self.warning_counter = 0
        if self.shuffle:
            np.random.shuffle(self.idxs)
