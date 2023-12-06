"""
Class to compute the impact function.
"""

from .impact import Impact

import hashlib
import pickle
import keras
import tensorflow as tf
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path


class ImpactDeepLearning(Impact):
    """
    The Deep Learning Impact class.

    Parameters
    ----------
    events: Events
        The events object.
    target_type: str
        The target type. Options are: 'occurrence', 'damage_ratio'
    random_state: int|None
        The random state to use for the random number generator.
        Default: 42. Set to None to not set the random seed.
    reload_trained_models: bool
        Whether to reload the previously trained models or not.
    """

    class DataGenerator(keras.utils.Sequence):
        def __init__(self, dates, x_static, x_precip, x_dem, y, batch_size=32,
                     shuffle=True, load=False, window_size=12, mean_static=None,
                     std_static=None, mean_precip=None, std_precip=None, mean_dem=None,
                     std_dem=None, precip_normalization='domain-average', tmp_dir=None):
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
            mean_static: np.array
                The mean of the static data.
            std_static: np.array
                The standard deviation of the static data.
            mean_precip: np.array
                The mean of the precipitation data.
            std_precip: np.array
                The standard deviation of the precipitation data.
            mean_dem: np.array
                The mean of the DEM data.
            std_dem: np.array
                The standard deviation of the DEM data.
            precip_normalization: str
                The normalization to apply to the precipitation data: 'domain-average',
                or 'per-pixel'.
            tmp_dir: Path
                The temporary directory to use.
            """
            self.dates = dates
            self.y = y
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.window_size = window_size

            self.X_static = x_static
            self.X_precip = x_precip
            self.X_dem = x_dem
            self._compute_predictor_statistics(mean_dem, mean_precip, mean_static,
                                               precip_normalization, std_dem,
                                               std_precip, std_static, tmp_dir)

            # Normalize
            self.X_static = (self.X_static - self.mean_static) / self.std_static
            self.X_precip = (self.X_precip - self.mean_precip) / self.std_precip
            self.X_dem = (self.X_dem - self.mean_dem) / self.std_dem
            self.n_samples = self.y.shape[0]

            self.on_epoch_end()

            if load:
                print('Loading data into RAM')
                self.X_precip.load()
                self.X_dem.load()

        def _compute_predictor_statistics(self, mean_dem, mean_precip, mean_static,
                                          precip_normalization, std_dem, std_precip,
                                          std_static, tmp_dir):
            self.mean_static = np.mean(
                self.X_static, axis=0) if mean_static is None else mean_static
            self.std_static = np.std(
                self.X_static, axis=0) if std_static is None else std_static

            file_hash = self._compute_hash(precip_normalization)
            if precip_normalization == 'domain-average':
                # Compute the mean of the precipitation
                if mean_precip is None:
                    # If pickle file exists, load it
                    tmp_filename = tmp_dir / f'mean_precip_{file_hash}.pickle'
                    if tmp_filename.exists():
                        with open(tmp_filename, 'rb') as f:
                            self.mean_precip = pickle.load(f)
                    else:
                        self.mean_precip = self.X_precip.mean(
                            ('time', 'x', 'y')
                        ).compute()
                        # Save to pickle file
                        with open(tmp_filename, 'wb') as f:
                            pickle.dump(self.mean_precip, f)
                else:
                    self.mean_precip = mean_precip

                # Compute the standard deviation of the precipitation
                if std_precip is None:
                    # If pickle file exists, load it
                    tmp_filename = tmp_dir / f'std_precip_{file_hash}.pickle'
                    if tmp_filename.exists():
                        with open(tmp_filename, 'rb') as f:
                            self.std_precip = pickle.load(f)
                    else:
                        self.std_precip = self.X_precip.std('time').mean(
                            ('x', 'y')
                        ).compute()
                        # Save to pickle file
                        with open(tmp_filename, 'wb') as f:
                            pickle.dump(self.std_precip, f)

            elif precip_normalization == 'per-pixel':
                # Compute the mean of the precipitation
                if mean_precip is None:
                    # If pickle file exists, load it
                    tmp_filename = tmp_dir / f'mean_precip_{file_hash}.pickle'
                    if tmp_filename.exists():
                        with open(tmp_filename, 'rb') as f:
                            self.mean_precip = pickle.load(f)
                    else:
                        self.mean_precip = self.X_precip.mean('time').compute()
                        # Save to pickle file
                        with open(tmp_filename, 'wb') as f:
                            pickle.dump(self.mean_precip, f)
                else:
                    self.mean_precip = mean_precip

                    # Compute the standard deviation of the precipitation
                    if std_precip is None:
                        # If pickle file exists, load it
                        tmp_filename = tmp_dir / f'std_precip_{file_hash}.pickle'
                        if tmp_filename.exists():
                            with open(tmp_filename, 'rb') as f:
                                self.std_precip = pickle.load(f)
                        else:
                            self.std_precip = self.X_precip.std('time').compute()
                            # Save to pickle file
                            with open(tmp_filename, 'wb') as f:
                                pickle.dump(self.std_precip, f)
            else:
                raise ValueError(
                    f'Unknown normalization method: {precip_normalization}')

            self.mean_dem = self.X_dem.mean(
                ('x', 'y')).compute() if mean_dem is None else mean_dem
            self.std_dem = self.X_dem.std(
                ('x', 'y')).compute() if std_dem is None else std_dem

        def _compute_hash(self, precip_normalization):
            tag_data = (
                    pickle.dumps(precip_normalization) +
                    pickle.dumps(len(self.dates)) +
                    pickle.dumps(self.dates[0]) +
                    pickle.dumps(self.dates[-1]) +
                    pickle.dumps(self.X_precip.shape))

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

    def __init__(self, events, target_type='occurrence', random_state=42,
                 reload_trained_models=False):
        super().__init__(events, target_type=target_type, random_state=random_state)
        self.reload_trained_models = reload_trained_models

        self.precipitation = None
        self.dem = None
        self.precip_days_before = 5
        self.precip_days_after = 3

    def fit(self, tag=None):
        """
        Fit the model.

        Parameters
        ----------
        tag: str
            The tag to add to the file name.
        """
        start_train = self.dates_train[0] - pd.to_timedelta(
            self.precip_days_before, unit='D')
        end_train = self.dates_train[-1] + pd.to_timedelta(
            self.precip_days_after, unit='D')
        dg_train = self.DataGenerator(
            dates=self.dates_train,
            x_static=self.x_train,
            x_precip=self.precipitation.sel(time=slice(start_train, end_train)),
            x_dem=self.dem,
            y=self.y_train,
            batch_size=32,
            shuffle=True,
            load=True,
            window_size=12,
            tmp_dir=self.tmp_dir,
        )

        start_val = self.dates_valid[0] - pd.to_timedelta(
            self.precip_days_before, unit='D')
        end_val = self.dates_valid[-1] + pd.to_timedelta(
            self.precip_days_after, unit='D')
        dg_val = self.DataGenerator(
            dates=self.dates_valid,
            x_static=self.x_valid,
            x_precip=self.precipitation.sel(time=slice(start_val, end_val)),
            x_dem=self.dem,
            y=self.y_valid,
            batch_size=32,
            shuffle=False,
            load=True,
            window_size=12,
            mean_static=dg_train.mean_static,
            std_static=dg_train.std_static,
            mean_precip=dg_train.mean_precip,
            std_precip=dg_train.std_precip,
            mean_dem=dg_train.mean_dem,
            std_dem=dg_train.std_dem,
        )

    def _define_model(self):
        """
        Define the model.
        """
        raise NotImplementedError

    def _create_model_tmp_file_name(self):
        """
        Create the temporary file name for the model.
        """
        tag_model = (
                pickle.dumps(self.df.shape) + pickle.dumps(self.df.columns) +
                pickle.dumps(self.df.iloc[0]) + pickle.dumps(self.features) +
                pickle.dumps(self.class_weight) + pickle.dumps(self.random_state) +
                pickle.dumps(self.target_type))
        model_hashed_name = f'dl_model_{hashlib.md5(tag_model).hexdigest()}.pickle'
        tmp_filename = self.tmp_dir / model_hashed_name

        return tmp_filename

    def set_precipitation(self, precipitation):
        """
        Set the precipitation data.

        Parameters
        ----------
        precipitation: xarray.Dataset
            The precipitation data.
        """
        assert len(precipitation.dims) == 3, "Precipitation must be 3D"
        if self.dem is not None:
            assert precipitation['precip'].shape[1:] == self.dem.shape, \
                "DEM and precipitation must have the same shape"
        self.precipitation = precipitation
        self.precipitation['time'] = pd.to_datetime(self.precipitation['time'])

    def set_dem(self, dem):
        """
        Set the DEM data.

        Parameters
        ----------
        dem: xarray.Dataset
            The DEM data.
        """
        assert dem.ndim == 2, "DEM must be 2D"
        if self.precipitation is not None:
            assert dem.shape == self.precipitation.isel(time=0).shape, \
                "DEM and precipitation must have the same shape"
        self.dem = dem
