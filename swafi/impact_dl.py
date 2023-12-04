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
        def __init__(self, x_static, x_precip, x_dem, y, batch_size=32, shuffle=True,
                     load=True, window_size=12, mean_static=None, std_static=None,
                     mean_precip=None, std_precip=None, mean_dem=None, std_dem=None)
            """
            Data generator class.
            Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
            Adapted by https://github.com/pangeo-data/WeatherBench/blob/master/src/train_nn.py

            Parameters
            ----------
            x_static: np.array
                The static predictor variables (0D).
            x_precip: xarray.DataArray
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
            """
            self.y = y
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.window_size = window_size

            self.X_static = x_static
            self.X_precip = x_precip
            self.X_dem = x_dem
            self.mean_static = self.X_static.mean(
                ('events')).compute() if mean_static is None else mean_static
            self.std_static = self.X_static.std('time').mean(
                ('lat', 'lon')).compute() if std_static is None else std_static
            self.mean_precip = self.X_precip.mean(
                ('time', 'x', 'y')).compute() if mean_precip is None else mean_precip
            self.std_precip = self.X_precip.std('time').mean(
                ('x', 'y')).compute() if std_precip is None else std_precip
            self.mean_dem = self.X_dem.mean(
                ('x', 'y')).compute() if mean_dem is None else mean_dem
            self.std_dem = self.X_dem.std(
                ('x', 'y')).compute() if std_dem is None else std_dem

            # Normalize
            self.X_static = (self.X_static - self.mean_static) / self.std_static
            self.X_precip = (self.X_precip - self.mean_precip) / self.std_precip
            self.X_dem = (self.X_dem - self.mean_dem) / self.std_dem
            self.n_samples = self.y.shape[0]

            self.on_epoch_end()

            if load:
                print('Loading data into RAM')
                self.X_static.load()
                self.X_precip.load()
                self.X_dem.load()
                self.y.load()

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

    def fit(self, tag=None):
        """
        Fit the model.

        Parameters
        ----------
        tag: str
            The tag to add to the file name.
        """
        raise NotImplementedError

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
