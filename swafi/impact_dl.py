"""
Class to compute the impact function.
"""

from .impact import Impact
from .utils.data_generator import DataGenerator

import hashlib
import pickle
import keras
import tensorflow as tf
import pandas as pd


class DeepImpact(tf.keras.Model):
    """
    Model factory.

    Parameters
    ----------
    task: str
        The task. Options are: 'regression', 'classification'
    input_2d_size: list
        The input 2D size.
    input_1d_size: list
        The input 1D size.
    dropout_rate: float
        The dropout rate.
    with_spatial_dropout: bool
        Whether to use spatial dropout or not.
    with_batchnorm: bool
        Whether to use batch normalization or not.
    inner_activation: str
        The inner activation function.
    """
    def __init__(self, task, input_2d_size, input_1d_size, dropout_rate=0.2,
                 with_spatial_dropout=True, with_batchnorm=True, inner_activation='relu'):
        super(DeepImpact, self).__init__()
        self.model = None
        self.task = task
        self.input_2d_size = list(input_2d_size)
        self.input_1d_size = list(input_1d_size)
        self.dropout_rate = dropout_rate
        self.with_spatial_dropout = with_spatial_dropout
        self.with_batchnorm = with_batchnorm
        self.inner_activation = inner_activation
        self.last_activation = 'relu' if task == 'regression' else 'sigmoid'

        self.build_model()

    def build_model(self):
        """
        Build the model.
        """
        input_2d = keras.layers.Input(shape=self.input_2d_size, name='input_2d')
        input_1d = keras.layers.Input(shape=self.input_1d_size, name='input_1d')

        # 2D convolution
        x = self.conv2d_block(input_2d, filters=32, kernel_size=3)
        x = self.conv2d_block(x, filters=64, kernel_size=3)
        x = self.conv2d_block(x, filters=128, kernel_size=3)

        # Flatten
        x = keras.layers.Flatten()(x)

        # Concatenate with 1D input
        x = keras.layers.concatenate([x, input_1d])

        # Fully connected
        x = keras.layers.Dense(256, activation=self.inner_activation)(x)
        x = keras.layers.Dense(128, activation=self.inner_activation)(x)
        x = keras.layers.Dense(64, activation=self.inner_activation)(x)
        x = keras.layers.Dense(32, activation=self.inner_activation)(x)

        # Last activation
        x = keras.layers.Dense(1, activation=self.last_activation)(x)

        self.model = keras.models.Model(inputs=[input_2d, input_1d], outputs=x)

    def conv2d_block(self, x, filters, kernel_size=3, initializer='he_normal',
                     activation='he_normal'):
        """
        Convolution block.

        Parameters
        ----------
        x: keras.layers.Layer
            The input layer.
        filters: int
            The number of filters.
        kernel_size: int
            The kernel size.
        initializer: str
            The initializer.
        activation: str
            The activation function.

        Returns
        -------
        The output layer.
        """
        if activation == 'default':
            activation = self.inner_activation

        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation=activation,
            kernel_initializer=initializer,
        )(x)
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation=activation,
        )(x)

        if self.with_batchnorm:
            x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
        )(x)

        if self.with_spatial_dropout and self.dropout_rate > 0:
            x = keras.layers.SpatialDropout2D(
                rate=self.dropout_rate,
            )(x)
        elif self.dropout_rate > 0:
            x = keras.layers.Dropout(
                rate=self.dropout_rate,
            )(x)

        return x

    def call(self, inputs, **kwargs):
        """
        Call the model.

        Parameters
        ----------
        inputs: list
            The inputs.
        **kwargs

        Returns
        -------
        The output.
        """
        return self.model(inputs, kwargs)


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

    def __init__(self, events, target_type='occurrence', random_state=42,
                 reload_trained_models=False):
        super().__init__(events, target_type=target_type, random_state=random_state)
        self.reload_trained_models = reload_trained_models

        self.precipitation = None
        self.dem = None

        # Options
        self.precip_days_before = 8
        self.precip_days_after = 3
        self.precip_window_size = 12
        self.transform_static = 'standardize'  # 'standardize' or 'normalize'
        self.transform_2d = 'standardize'  # 'standardize' or 'normalize'
        self.precip_trans_domain = 'domain-average'  # 'domain-average' or 'per-pixel'

        # Hyperparameters
        self.batch_size = 32
        self.epochs = 100

    def fit(self, tag=None):
        """
        Fit the model.

        Parameters
        ----------
        tag: str
            The tag to add to the file name.
        """
        self._define_model(input_2d_size=[self.precip_window_size,
                                          self.precip_window_size],
                           input_1d_size=self.x_train.shape[1:])

        start_train = self.dates_train[0] - pd.to_timedelta(
            self.precip_days_before, unit='D')
        end_train = self.dates_train[-1] + pd.to_timedelta(
            self.precip_days_after, unit='D')
        dg_train = DataGenerator(
            dates=self.dates_train,
            x_static=self.x_train,
            x_precip=self.precipitation.sel(time=slice(start_train, end_train)),
            x_dem=self.dem,
            y=self.y_train,
            batch_size=self.batch_size,
            shuffle=True,
            window_size=self.precip_window_size,
            tmp_dir=self.tmp_dir,
            transform_static=self.transform_static,
            transform_2d=self.transform_2d,
            precip_transformation_domain=self.precip_trans_domain,
            log_transform_precip=True
        )

        start_val = self.dates_valid[0] - pd.to_timedelta(
            self.precip_days_before, unit='D')
        end_val = self.dates_valid[-1] + pd.to_timedelta(
            self.precip_days_after, unit='D')
        dg_val = DataGenerator(
            dates=self.dates_valid,
            x_static=self.x_valid,
            x_precip=self.precipitation.sel(time=slice(start_val, end_val)),
            x_dem=self.dem,
            y=self.y_valid,
            batch_size=self.batch_size,
            shuffle=True,
            window_size=self.precip_window_size,
            transform_static=self.transform_static,
            transform_2d=self.transform_2d,
            precip_transformation_domain=self.precip_trans_domain,
            log_transform_precip=True,
            mean_static=dg_train.mean_static,
            std_static=dg_train.std_static,
            mean_precip=dg_train.mean_precip,
            std_precip=dg_train.std_precip,
            min_static=dg_train.min_static,
            max_static=dg_train.max_static,
            max_precip=dg_train.max_precip
        )

    def _define_model(self, input_2d_size, input_1d_size):
        """
        Define the model.
        """
        self.model = DeepImpact(
            task=self.target_type,
            input_2d_size=input_2d_size,
            input_1d_size=input_1d_size,
            dropout_rate=0.2,
            with_spatial_dropout=True,
            with_batchnorm=True,
            inner_activation='relu'
        )

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
