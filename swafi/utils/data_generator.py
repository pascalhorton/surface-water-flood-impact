"""
Class to compute the impact function.
"""

import hashlib
import pickle
import keras
import numpy as np
from pathlib import Path


class DataGenerator(keras.utils.Sequence):
    def __init__(self, event_props, x_static, x_precip, x_dem, y, batch_size=32,
                 shuffle=True, load_dump_precip_data=True, precip_window_size=12,
                 precip_grid_resol=1000, precip_days_before=8, precip_days_after=3,
                 tmp_dir=None, transform_static='standardize',
                 transform_2d='standardize',
                 precip_transformation_domain='domain-average',
                 log_transform_precip=True, mean_static=None, std_static=None,
                 mean_precip=None, std_precip=None, min_static=None,
                 max_static=None, max_precip=None):
        """
        Data generator class.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Adapted by https://github.com/pangeo-data/WeatherBench/blob/master/src/train_nn.py

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
        load_dump_precip_data: bool
            Whether to load the full data into memory and dump it to a pickle file.
        precip_window_size: int
            The window size for the 2D predictors [km].
        precip_grid_resol: int
            The grid resolution of the precipitation data [m].
        precip_days_before: int
            The number of days before the event to include in the 2D predictors.
        precip_days_after: int
            The number of days after the event to include in the 2D predictors.
        tmp_dir: Path
            The temporary directory to use.
        transform_static: str
            The transformation to apply to the static data: 'standardize' or
            'normalize'.
        transform_2d: str
            The transformation to apply to the 2D data: 'standardize' or
            'normalize'.
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
        max_precip: np.array
            The max of the precipitation data.
        """
        self.event_props = event_props
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.precip_window_size = precip_window_size
        self.precip_grid_resol = precip_grid_resol
        self.precip_days_before = precip_days_before
        self.precip_days_after = precip_days_after

        self.mean_static = mean_static
        self.std_static = std_static
        self.mean_precip = mean_precip
        self.std_precip = std_precip
        self.min_static = min_static
        self.max_static = max_static
        self.max_precip = max_precip
        self.mean_dem = None
        self.std_dem = None
        self.min_dem = None
        self.max_dem = None

        self.X_static = x_static
        self.X_precip = x_precip
        self.X_dem = x_dem
        self._compute_predictor_statistics(
            transform_static, transform_2d, precip_transformation_domain,
            log_transform_precip, tmp_dir)
        if transform_static == 'standardize':
            self._standardize_inputs()
        elif transform_static == 'normalize':
            self._normalize_inputs()

        self._restrict_spatial_domain()
        self._restrict_temporal_selection()

        self.n_samples = self.y.shape[0]
        self.idxs = np.arange(self.n_samples)

        self.on_epoch_end()

        self.X_dem.load()
        if load_dump_precip_data:
            print('Loading data into RAM')
            self._load_dump_precip_data(transform_2d, precip_transformation_domain,
                                        log_transform_precip, tmp_dir)

    def get_channels_nb(self):
        input_2d_channels = 0
        if self.X_precip is not None:
            input_2d_channels += self.precip_days_after + self.precip_days_before
            input_2d_channels *= 24  # Hourly time step
            input_2d_channels += 1
        if self.X_dem is not None:
            input_2d_channels += 1  # Add one for the DEM layer

        return input_2d_channels

    def _standardize_inputs(self):
        self.X_static = (self.X_static - self.mean_static) / self.std_static
        self.X_precip['precip'] = ((self.X_precip['precip'] - self.mean_precip) /
                                   self.std_precip)
        self.X_dem = (self.X_dem - self.mean_dem) / self.std_dem

    def _normalize_inputs(self):
        self.X_static = ((self.X_static - self.min_static) /
                         (self.max_static - self.min_static))
        self.X_precip['precip'] = self.X_precip['precip'] / self.max_precip
        self.X_dem = (self.X_dem - self.min_dem) / (self.max_dem - self.min_dem)

    def _load_dump_precip_data(self, transform_2d, precip_transformation_domain,
                               log_transform_precip, tmp_dir):
        """ Load all the precipitation data into memory. """
        if tmp_dir is None:
            raise ValueError('tmp_dir must be specified')

        file_hash = self._compute_hash_precip(
            transform_2d, precip_transformation_domain, log_transform_precip)
        file_precip = tmp_dir / f'precip_{file_hash}.pickle'

        # If pickle file exists, load it
        if file_precip.exists():
            with open(file_precip, 'rb') as f:
                self.X_precip = pickle.load(f)
            return

        # Otherwise, load the data and save it to a pickle file
        self.X_precip.load()
        with open(file_precip, 'wb') as f:
            pickle.dump(self.X_precip, f)

    def _restrict_spatial_domain(self):
        """ Restrict the spatial domain of the precipitation and DEM data. """
        precip_window_size_m = self.precip_window_size * self.precip_grid_resol
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
            t_min = self.event_props[:, 0].min() - np.timedelta64(
                self.precip_days_before, 'D')
            t_max = self.event_props[:, 0].max() + np.timedelta64(
                self.precip_days_after, 'D')
            self.X_precip = self.X_precip.sel(
                time=slice(t_min, t_max)
            )

    def _compute_predictor_statistics(self, transform_static, transform_2d,
                                      precip_transformation_domain,
                                      log_transform_precip, tmp_dir):
        print('Computing/assigning static predictor statistics')
        if transform_static == 'standardize':
            # Compute the mean and standard deviation of the static data
            if self.mean_static is None:
                self.mean_static = np.mean(self.X_static, axis=0)
            if self.std_static is None:
                self.std_static = np.std(self.X_static, axis=0)
        elif transform_static == 'normalize':
            # Compute the min and max of the static data
            if self.min_static is None:
                self.min_static = np.min(self.X_static, axis=0)
            if self.max_static is None:
                self.max_static = np.max(self.X_static, axis=0)

        print('Computing DEM predictor statistics')
        if transform_2d == 'standardize':
            # Compute the mean and standard deviation of the DEM (non-temporal)
            self.mean_dem = self.X_dem.mean(('x', 'y')).compute().values
            self.std_dem = self.X_dem.std(('x', 'y')).compute().values
        elif transform_2d == 'normalize':
            # Compute the min and max of the DEM (non-temporal)
            self.min_dem = self.X_dem.min(('x', 'y')).compute().values
            self.max_dem = self.X_dem.max(('x', 'y')).compute().values

        # Log transform the precipitation (log(1 + x))
        if log_transform_precip:
            print('Log-transforming precipitation')
            self.X_precip['precip'] = np.log1p(self.X_precip['precip'])

        # Compute the mean and standard deviation of the precipitation
        print('Computing/assigning precipitation predictor statistics')
        if (transform_2d == 'standardize' and self.mean_precip is not None
                and self.std_precip is not None):
            return

        if transform_2d == 'normalize' and self.max_precip is not None:
            return

        # If pickle file exists, load it
        file_hash = self._compute_hash(precip_transformation_domain,
                                       log_transform_precip)
        file_mean_precip = tmp_dir / f'mean_precip_{file_hash}.pickle'
        file_std_precip = tmp_dir / f'std_precip_{file_hash}.pickle'
        file_max_precip = tmp_dir / f'max_precip_{file_hash}.pickle'

        if (transform_2d == 'standardize' and file_mean_precip.exists()
                and file_std_precip.exists()):
            with open(file_mean_precip, 'rb') as f:
                self.mean_precip = pickle.load(f)
            with open(file_std_precip, 'rb') as f:
                self.std_precip = pickle.load(f)
            return

        if transform_2d == 'normalize' and file_max_precip.exists():
            with open(file_max_precip, 'rb') as f:
                self.max_precip = pickle.load(f)
            return

        if precip_transformation_domain == 'domain-average':
            if transform_2d == 'standardize':
                self.mean_precip = self.X_precip['precip'].mean(
                    ('time', 'x', 'y')).compute().values
                self.std_precip = self.X_precip['precip'].std('time').mean(
                    ('x', 'y')).compute().values
            elif transform_2d == 'normalize':
                self.max_precip = self.X_precip['precip'].max(
                    ('time', 'x', 'y')).compute().values
        elif precip_transformation_domain == 'per-pixel':
            if transform_2d == 'standardize':
                self.mean_precip = self.X_precip['precip'].mean(
                    'time').compute().values
                self.std_precip = self.X_precip['precip'].std(
                    'time').compute().values
            elif transform_2d == 'normalize':
                self.max_precip = self.X_precip['precip'].max(
                    'time').compute().values
        else:
            raise ValueError(
                f'Unknown option: {precip_transformation_domain}')

        # Save to pickle file
        if transform_2d == 'standardize':
            with open(file_mean_precip, 'wb') as f:
                pickle.dump(self.mean_precip, f)
            with open(file_std_precip, 'wb') as f:
                pickle.dump(self.std_precip, f)
        elif transform_2d == 'normalize':
            with open(file_max_precip, 'wb') as f:
                pickle.dump(self.max_precip, f)

    def _compute_hash(self, precip_transformation_domain, log_transform_precip):
        tag_data = (
                pickle.dumps(precip_transformation_domain) +
                pickle.dumps(log_transform_precip) +
                pickle.dumps(len(self.event_props[:, 0])) +
                pickle.dumps(self.event_props[0, 0]) +
                pickle.dumps(self.event_props[-1, 0]) +
                pickle.dumps(self.X_precip['precip'].shape))

        return hashlib.md5(tag_data).hexdigest()

    def _compute_hash_precip(self, transform_2d, precip_transformation_domain,
                             log_transform_precip):
        tag_data = (
                pickle.dumps(transform_2d) +
                pickle.dumps(precip_transformation_domain) +
                pickle.dumps(log_transform_precip) +
                pickle.dumps(self.X_precip['x']) +
                pickle.dumps(self.X_precip['y']) +
                pickle.dumps(self.X_precip['time'][0]) +
                pickle.dumps(self.X_precip['time'][-1]) +
                pickle.dumps(self.X_precip['precip'].shape))

        return hashlib.md5(tag_data).hexdigest()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        """Generate one batch of data"""
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]

        # Select the events
        y = self.y[idxs]
        event_props = self.event_props[idxs]

        # Select the corresponding static data
        x_static = self.X_static[idxs, :]

        # Select the 2D data
        x_2d = np.zeros((self.batch_size,
                         self.precip_window_size,
                         self.precip_window_size,
                         self.get_channels_nb()))

        precip_window_size_m = self.precip_window_size * self.precip_grid_resol

        for i_batch, event in enumerate(event_props):
            # Temporal selection
            t_start = event[0] - np.timedelta64(self.precip_days_before, 'D')
            t_end = event[0] + np.timedelta64(self.precip_days_after, 'D')

            # Spatial domain
            x_start = event[1] - precip_window_size_m / 2
            x_end = event[1] + precip_window_size_m / 2 - self.precip_grid_resol
            y_start = event[2] + precip_window_size_m / 2
            y_end = event[2] - precip_window_size_m / 2 + self.precip_grid_resol

            # Select the corresponding precipitation data (5 days prior the event)
            x_precip_ev = self.X_precip['precip'].sel(
                time=slice(t_start, t_end),
                x=slice(x_start, x_end),
                y=slice(y_start, y_end)
            ).to_numpy()

            # Extract the precipitation data
            x_precip_ev = np.moveaxis(x_precip_ev, 0, -1)

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
            x_2d_ev = np.concatenate([x_precip_ev, x_dem_ev], axis=-1)
            if x_2d_ev.shape[2] != self.get_channels_nb():
                print(f"Shape mismatch: {x_2d_ev.shape[2]} != {self.get_channels_nb()}")
                print(f"Event: {event}")
                x_2d[i_batch] = np.zeros((self.precip_window_size,
                                          self.precip_window_size,
                                          self.get_channels_nb()))
                y[i_batch] = 0
                continue

            x_2d[i_batch] = x_2d_ev

        return [x_2d, x_static], y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.idxs = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.idxs)
