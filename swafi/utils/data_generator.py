"""
Class to compute the impact function.
"""

import hashlib
import pickle
import keras
import numpy as np
from pathlib import Path


class DataGenerator(keras.utils.Sequence):
    def __init__(self, dates, x_static, x_precip, x_dem, y, batch_size=32,
                 shuffle=True, load=False, window_size=12, tmp_dir=None,
                 transform_static='standardize', transform_2d='standardize',
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
        dates: np.array
            The dates of the events.
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
        load: bool
            Whether to load the data into memory or not.
        window_size: int
            The window size for the 2D predictors [km].
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
        self.dates = dates
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.window_size = window_size

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

        self.n_samples = self.y.shape[0]

        self.on_epoch_end()

        if load:
            print('Loading data into RAM')
            self.X_precip.load()
            self.X_dem.load()

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
                pickle.dumps(len(self.dates)) +
                pickle.dumps(self.dates[0]) +
                pickle.dumps(self.dates[-1]) +
                pickle.dumps(self.X_precip['precip'].shape))

        return hashlib.md5(tag_data).hexdigest()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        """Generate one batch of data"""
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]

        # Select the events
        events = self.y.isel(time=idxs)
        y = events.values

        # Select the corresponding static data
        X_static = self.X_static.isel(events=idxs).values

        X_2d = np.zeros((len(events), self.window_size, self.window_size, 5))

        for event in events:
            # Select the corresponding precipitation data (5 days prior the event)
            X_precip_ev = self.X_precip.sel(
                time=slice(event.time - np.timedelta64(5, 'D'),
                           event.time + np.timedelta64(23, 'H')),
                x=slice(event.x - self.window_size / 2,
                        event.x + self.window_size / 2),
                y=slice(event.y - self.window_size / 2,
                        event.y + self.window_size / 2)
            ).values

            # Select the corresponding DEM data in the window
            X_dem_ev = self.X_dem.sel(
                x=slice(event.x - self.window_size / 2,
                        event.x + self.window_size / 2),
                y=slice(event.y - self.window_size / 2,
                        event.y + self.window_size / 2)
            ).values

            # Normalize
            X_precip_ev = (X_precip_ev - self.mean_precip) / self.std_precip
            X_dem_ev = (X_dem_ev - self.mean_dem) / self.std_dem

            # Concatenate
            X_2d_ev = np.concatenate([X_precip_ev, X_dem_ev], axis=-1)
            X_2d[i] = X_2d_ev

        return X_static, X_2d, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)