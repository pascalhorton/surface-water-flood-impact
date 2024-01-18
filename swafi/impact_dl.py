"""
Class to compute the impact function.
"""

from .impact import Impact
from .model_dl import DeepImpact
from .utils.data_generator import DataGenerator
from .utils.verification import compute_confusion_matrix, print_classic_scores, \
    assess_roc_auc, compute_score_binary

import hashlib
import pickle
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime


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
        self.dg_train = None
        self.dg_val = None
        self.dg_test = None

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
        self._create_data_generator_train()
        self._create_data_generator_valid()

        # Define the model
        self._define_model(input_2d_size=[self.precip_window_size,
                                          self.precip_window_size,
                                          self.dg_train.get_channels_nb()],
                           input_1d_size=self.x_train.shape[1:])

        # Early stopping
        callback = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)

        # Clear session and set the seed
        keras.backend.clear_session()
        keras.utils.set_random_seed(42)

        # Define the optimizer
        optimizer = self._define_optimizer(
            n_samples=len(self.dg_train), lr_method='constant', lr=0.001)

        # Get loss function
        loss_fn = self._get_loss_function()

        # Compile the model
        self.model.compile(
            loss=loss_fn,
            optimizer=optimizer,
            metrics=['accuracy']
        )

        # Print the model summary
        print("Model summary:")
        print(self.model.model.summary())

        # Fit the model
        hist = self.model.fit(
            self.dg_train,
            epochs=self.epochs,
            validation_data=self.dg_val,
            callbacks=[callback],
            verbose=1
        )

        # Plot the training history
        self._plot_training_history(hist)

    def assess_model_on_all_periods(self):
        """
        Assess the model on all periods.
        """
        self._create_data_generator_test()
        self._assess_model_dg(self.dg_train, 'Train period')
        self._assess_model_dg(self.dg_val, 'Validation period')
        self._assess_model_dg(self.dg_test, 'Test period')

    def _assess_model_dg(self, dg, period_name):
        """
        Assess the model on a single period.
        """
        if self.model is None:
            raise ValueError("Model not defined")

        x, y = dg.get_all_data()
        y_pred = self.model.predict(x)

        print(f"\nSplit: {period_name}")

        # Compute the scores
        if self.target_type == 'occurrence':
            y_pred_class = (y_pred > 0.5).astype(int)
            tp, tn, fp, fn = compute_confusion_matrix(y, y_pred_class)
            print_classic_scores(tp, tn, fp, fn)
            assess_roc_auc(y, y_pred[:, 1])
        else:
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            print(f"RMSE: {rmse}")
        print(f"----------------------------------------")

    def _get_loss_function(self):
        """
        Get the loss function.

        Returns
        -------
        The loss function.
        """
        if self.target_type == 'occurrence':
            if self.class_weight is None:
                loss_fn = 'binary_crossentropy'
            else:
                # Set class weights as float32
                class_weight = self.class_weight.copy()
                for key in class_weight:
                    class_weight[key] = float(class_weight[key])
                loss_fn = self._weighted_binary_cross_entropy(
                    weights=class_weight)
        else:
            loss_fn = 'mse'

        return loss_fn

    @staticmethod
    def _weighted_binary_cross_entropy(weights, from_logits=False):
        """
        Weighted binary cross entropy.

        Parameters
        ----------
        weights: dict
            The weights.
        from_logits: bool
            Whether the input is logit or not.

        Returns
        -------
        The loss function.
        """

        def weighted_binary_cross_entropy(y_true, y_pred):
            """
            Weighted binary cross entropy.
            From: https://stackoverflow.com/questions/46009619/keras-weighted-binary-crossentropy

            Parameters
            ----------
            y_true: array-like
                The true values.
            y_pred: array-like
                The predicted values.

            Returns
            -------
            The loss.
            """
            tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
            tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)

            weights_v = tf.where(tf.equal(tf_y_true, 1), weights[1], weights[0])
            ce = keras.metrics.binary_crossentropy(
                tf_y_true, tf_y_pred, from_logits=from_logits)
            loss = tf.reduce_mean(tf.multiply(ce, weights_v))

            return loss

        return weighted_binary_cross_entropy

    def _create_data_generator_train(self):
        start_train = self.events_train[0, 0] - pd.to_timedelta(
            self.precip_days_before, unit='D')
        end_train = self.events_train[-1, 0] + pd.to_timedelta(
            self.precip_days_after, unit='D')
        self.dg_train = DataGenerator(
            event_props=self.events_train,
            x_static=self.x_train,
            x_precip=self.precipitation.sel(time=slice(start_train, end_train)),
            x_dem=self.dem,
            y=self.y_train,
            batch_size=self.batch_size,
            shuffle=True,
            precip_window_size=self.precip_window_size,
            precip_days_before=self.precip_days_before,
            precip_days_after=self.precip_days_after,
            tmp_dir=self.tmp_dir,
            transform_static=self.transform_static,
            transform_2d=self.transform_2d,
            precip_transformation_domain=self.precip_trans_domain,
            log_transform_precip=True
        )

    def _create_data_generator_valid(self):
        start_val = self.events_valid[0, 0] - pd.to_timedelta(
            self.precip_days_before, unit='D')
        end_val = self.events_valid[-1, 0] + pd.to_timedelta(
            self.precip_days_after, unit='D')
        self.dg_val = DataGenerator(
            event_props=self.events_valid,
            x_static=self.x_valid,
            x_precip=self.precipitation.sel(time=slice(start_val, end_val)),
            x_dem=self.dem,
            y=self.y_valid,
            batch_size=self.batch_size,
            shuffle=True,
            precip_window_size=self.precip_window_size,
            precip_days_before=self.precip_days_before,
            precip_days_after=self.precip_days_after,
            tmp_dir=self.tmp_dir,
            transform_static=self.transform_static,
            transform_2d=self.transform_2d,
            precip_transformation_domain=self.precip_trans_domain,
            log_transform_precip=True,
            mean_static=self.dg_train.mean_static,
            std_static=self.dg_train.std_static,
            mean_precip=self.dg_train.mean_precip,
            std_precip=self.dg_train.std_precip,
            min_static=self.dg_train.min_static,
            max_static=self.dg_train.max_static,
            max_precip=self.dg_train.max_precip
        )

    def _create_data_generator_test(self):
        start_test = self.events_test[0, 0] - pd.to_timedelta(
            self.precip_days_before, unit='D')
        end_test = self.events_test[-1, 0] + pd.to_timedelta(
            self.precip_days_after, unit='D')
        self.dg_test = DataGenerator(
            event_props=self.events_test,
            x_static=self.x_test,
            x_precip=self.precipitation.sel(time=slice(start_test, end_test)),
            x_dem=self.dem,
            y=self.y_test,
            batch_size=self.batch_size,
            shuffle=True,
            precip_window_size=self.precip_window_size,
            precip_days_before=self.precip_days_before,
            precip_days_after=self.precip_days_after,
            tmp_dir=self.tmp_dir,
            transform_static=self.transform_static,
            transform_2d=self.transform_2d,
            precip_transformation_domain=self.precip_trans_domain,
            log_transform_precip=True,
            mean_static=self.dg_train.mean_static,
            std_static=self.dg_train.std_static,
            mean_precip=self.dg_train.mean_precip,
            std_precip=self.dg_train.std_precip,
            min_static=self.dg_train.min_static,
            max_static=self.dg_train.max_static,
            max_precip=self.dg_train.max_precip
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

    def _define_optimizer(self, n_samples, lr_method='constant', lr=.001, init_lr=0.01):
        """
        Define the optimizer.

        Parameters
        ----------
        n_samples: int
            The number of samples.
        lr_method: str
            The learning rate method. Options are: 'cosine_decay', 'constant'
        lr: float
            The learning rate.
        init_lr: float
            The initial learning rate (for the cyclical and cosine_decay options).

        Returns
        -------
        The optimizer.
        """
        if lr_method == 'cosine_decay':
            decay_steps = self.epochs * (n_samples / self.batch_size)
            lr_decayed_fn = keras.optimizers.schedules.CosineDecay(
                init_lr, decay_steps)
            optimizer = keras.optimizers.Adam(lr_decayed_fn)
        elif lr_method == 'constant':
            optimizer = keras.optimizers.Adam(learning_rate=lr)
        else:
            raise ValueError('learning rate schedule not well defined.')

        return optimizer

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

    @staticmethod
    def _plot_training_history(hist):
        """
        Plot the training history.

        Parameters
        ----------
        hist: keras.callbacks.History
            The history.
        """
        now = datetime.datetime.now()

        plt.figure(figsize=(10, 5))
        plt.plot(hist.history['loss'], label='train')
        plt.plot(hist.history['val_loss'], label='valid')
        plt.legend()
        plt.title('Loss')
        plt.tight_layout()
        plt.savefig(f'loss_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(hist.history['accuracy'], label='train')
        plt.plot(hist.history['val_accuracy'], label='valid')
        plt.legend()
        plt.title('Accuracy')
        plt.tight_layout()
        plt.savefig(f'accuracy_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
        plt.show()
