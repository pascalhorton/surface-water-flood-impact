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
import dask

epsilon = 1e-7  # a small constant to avoid division by zero


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
    use_precip: bool
        Whether to use precipitation data (CombiPrecip) or not.
    precip_window_size: int
        The precipitation window size [km].
    precip_resolution: int
        The precipitation resolution [km].
    precip_time_step: int
        The precipitation time step [h].
    precip_days_before: int
        The number of days before the event to use for the precipitation.
    precip_days_after: int
        The number of days after the event to use for the precipitation.
    batch_size: int
        The batch size.
    epochs: int
        The number of epochs.
    """

    def __init__(self, events, target_type='occurrence', random_state=42,
                 reload_trained_models=False, use_precip=True, precip_window_size=8,
                 precip_resolution=1, precip_time_step=1, precip_days_before=8,
                 precip_days_after=3, batch_size=32, epochs=100):
        super().__init__(events, target_type=target_type, random_state=random_state)
        self.reload_trained_models = reload_trained_models

        self.precipitation = None
        self.dem = None
        self.dg_train = None
        self.dg_val = None
        self.dg_test = None

        # Check the precipitation parameters
        assert precip_window_size % precip_resolution == 0, \
            "precip_window_size must be divisible by precip_resolution"
        pixels_per_side = precip_window_size // precip_resolution
        assert pixels_per_side % 2 == 0, "pixels per side must be even"
        assert precip_days_before >= 0, "precip_days_before must be >= 0"
        assert precip_days_after >= 0, "precip_days_after must be >= 0"

        # Display if using GPU or CPU
        print("Built with CUDA: ", tf.test.is_built_with_cuda())
        print("Available GPU: ", tf.config.list_physical_devices('GPU'))

        # Options
        self.random_state = random_state
        self.use_precip = use_precip
        self.precip_days_before = precip_days_before
        self.precip_days_after = precip_days_after
        self.precip_window_size = precip_window_size
        self.precip_resolution = precip_resolution
        self.precip_time_step = precip_time_step
        self.transform_static = 'standardize'  # 'standardize' or 'normalize'
        self.transform_2d = 'standardize'  # 'standardize' or 'normalize'
        self.precip_trans_domain = 'per-pixel'  # 'domain-average' or 'per-pixel'
        self.factor_neg_reduction = 1

        # Hyperparameters
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, tag=None, dir_plots=None, show_plots=False):
        """
        Fit the model.

        Parameters
        ----------
        tag: str
            A tag to add to the file name.
        dir_plots: str
            The directory where to save the plots.
        show_plots: bool
            Whether to show the plots or not.
        """
        self._create_data_generator_train()
        self._create_data_generator_valid()

        # Define the model
        pixels_per_side = self.precip_window_size // self.precip_resolution
        if self.use_precip:
            self._define_model(input_2d_size=[pixels_per_side,
                                              pixels_per_side,
                                              self.dg_train.get_channels_nb()],
                               input_1d_size=self.x_train.shape[1:])
        else:
            self._define_model(input_2d_size=None,
                               input_1d_size=self.x_train.shape[1:])

        # Early stopping
        callback = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True)

        # Clear session and set the seed
        keras.backend.clear_session()
        keras.utils.set_random_seed(self.random_state)

        # Define the optimizer
        optimizer = self._define_optimizer(
            n_samples=len(self.dg_train), lr_method='constant', lr=0.001)

        # Get loss function
        loss_fn = self._get_loss_function()

        # Compile the model
        self.model.compile(
            loss=loss_fn,
            optimizer=optimizer,
            metrics=[self.csi]
        )

        # Print the model summary
        self.model.model.summary()

        # Fit the model
        verbose = 1 if show_plots else 2
        hist = self.model.fit(
            self.dg_train,
            epochs=self.epochs,
            validation_data=self.dg_val,
            callbacks=[callback],
            verbose=verbose
        )

        # Plot the training history
        self._plot_training_history(hist, dir_plots, show_plots, tag)

    def reduce_negatives_for_training(self, factor):
        """
        Reduce the number of negatives on the training set.

        Parameters
        ----------
        factor: float
            The factor to reduce the number of negatives.
        """
        self.factor_neg_reduction = factor

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

        n_batches = dg.get_number_of_batches_for_full_dataset()

        # Predict
        all_pred = []
        all_obs = []
        for i in range(n_batches):
            x, y = dg.get_ordered_batch_from_full_dataset(i)
            all_obs.append(y)
            y_pred_batch = self.model.predict(x, verbose=0)

            # Get rid of the single dimension
            y_pred_batch = y_pred_batch.squeeze()
            all_pred.append(y_pred_batch)

        # Concatenate predictions and obs from all batches
        y_pred = np.concatenate(all_pred, axis=0)
        y_obs = np.concatenate(all_obs, axis=0)

        print(f"\nSplit: {period_name}")

        # Compute the scores
        if self.target_type == 'occurrence':
            y_pred_class = (y_pred > 0.5).astype(int)
            tp, tn, fp, fn = compute_confusion_matrix(y_obs, y_pred_class)
            print_classic_scores(tp, tn, fp, fn)
            assess_roc_auc(y_obs, y_pred)
        else:
            rmse = np.sqrt(np.mean((y_obs - y_pred) ** 2))
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

    @staticmethod
    def csi(y_true, y_pred):
        """
        Compute the critical success index (CSI) for use in tensorflow.

        Parameters
        ----------
        y_true: array-like
            The true values.
        y_pred: array-like
            The predicted values.

        Returns
        -------
        The CSI score.
        """
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        y_pred = tf.cast(y_pred, dtype=y_pred.dtype)
        y_pred = tf.round(y_pred)  # convert probabilities to binary predictions
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        csi = tp / (tp + fp + fn + epsilon)

        return csi

    def _create_data_generator_train(self):
        self.dg_train = DataGenerator(
            event_props=self.events_train,
            x_static=self.x_train,
            x_precip=self.precipitation,
            x_dem=self.dem,
            y=self.y_train,
            batch_size=self.batch_size,
            shuffle=True,
            precip_window_size=self.precip_window_size,
            precip_resolution=self.precip_resolution,
            precip_time_step=self.precip_time_step,
            precip_days_before=self.precip_days_before,
            precip_days_after=self.precip_days_after,
            tmp_dir=self.tmp_dir,
            transform_static=self.transform_static,
            transform_2d=self.transform_2d,
            precip_transformation_domain=self.precip_trans_domain,
            log_transform_precip=True,
            debug=True
        )

        if self.factor_neg_reduction != 1:
            self.dg_train.reduce_negatives(self.factor_neg_reduction)

        if self.use_precip:
            self.dg_train.prepare_precip_data()

    def _create_data_generator_valid(self):
        self.dg_val = DataGenerator(
            event_props=self.events_valid,
            x_static=self.x_valid,
            x_precip=self.precipitation,
            x_dem=self.dem,
            y=self.y_valid,
            batch_size=self.batch_size,
            shuffle=True,
            precip_window_size=self.precip_window_size,
            precip_resolution=self.precip_resolution,
            precip_time_step=self.precip_time_step,
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
            max_precip=self.dg_train.max_precip,
            debug=True
        )

        if self.factor_neg_reduction != 1:
            self.dg_val.reduce_negatives(self.factor_neg_reduction)

        if self.use_precip:
            self.dg_val.prepare_precip_data()

    def _create_data_generator_test(self):
        self.dg_test = DataGenerator(
            event_props=self.events_test,
            x_static=self.x_test,
            x_precip=self.precipitation,
            x_dem=self.dem,
            y=self.y_test,
            batch_size=self.batch_size,
            shuffle=True,
            precip_window_size=self.precip_window_size,
            precip_resolution=self.precip_resolution,
            precip_time_step=self.precip_time_step,
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
            max_precip=self.dg_train.max_precip,
            debug=True
        )

        if self.use_precip:
            self.dg_test.prepare_precip_data()

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
            The number of samples. Used for the option 'cosine_decay'.
        lr_method: str
            The learning rate method. Options are: 'cosine_decay', 'constant'
        lr: float
            The learning rate. Used for the option 'constant'.
        init_lr: float
            The initial learning rate. Used for the option 'cosine_decay'.

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
        if not self.use_precip:
            print("Precipitation is not used and is therefore not loaded.")
            return

        assert len(precipitation.dims) == 3, "Precipitation must be 3D"

        hash_tag = self._compute_hash_precip_full_data(precipitation)
        filename = f"precip_full_{hash_tag}.pickle"
        tmp_filename = self.tmp_dir / filename

        if tmp_filename.exists():
            print("Precipitation already preloaded. Loading from pickle file.")
            with open(tmp_filename, 'rb') as f:
                precipitation = pickle.load(f)

        else:
            # Adapt the spatial resolution
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                if self.precip_resolution != 1:
                    precipitation = precipitation.coarsen(
                        x=self.precip_resolution,
                        y=self.precip_resolution,
                        boundary='trim'
                    ).mean()

            # Aggregate the precipitation at the desired time step
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                if self.precip_time_step != 1:
                    precipitation = precipitation.resample(
                        time=f'{self.precip_time_step}h',
                    ).sum(dim='time')

            # Save the precipitation
            with open(tmp_filename, 'wb') as f:
                pickle.dump(precipitation, f)

        # Check the shape of the precipitation and the DEM
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
        if not self.use_precip:
            print("DEM is not used and is therefore not loaded.")
            return

        assert dem.ndim == 2, "DEM must be 2D"

        # Adapt the spatial resolution
        if self.precip_resolution != 1:
            dem = dem.coarsen(
                x=self.precip_resolution, y=self.precip_resolution, boundary='trim'
            ).mean()

        if self.precipitation is not None:
            assert dem.shape == self.precipitation.isel(time=0).shape, \
                "DEM and precipitation must have the same shape"
        self.dem = dem

    def _compute_hash_precip_full_data(self, precipitation):
        tag_data = (
                pickle.dumps(self.precip_resolution) +
                pickle.dumps(self.precip_time_step) +
                pickle.dumps(precipitation['x']) +
                pickle.dumps(precipitation['y']) +
                pickle.dumps(precipitation['time'][0]) +
                pickle.dumps(precipitation['time'][-1]) +
                pickle.dumps(precipitation['precip'].shape))

        return hashlib.md5(tag_data).hexdigest()

    @staticmethod
    def _plot_training_history(hist, dir_plots, show_plots, prefix=None):
        """
        Plot the training history.

        Parameters
        ----------
        hist: keras.callbacks.History
            The history.
        dir_plots: str
            The directory where to save the plots.
        show_plots: bool
            Whether to show the plots or not.
        prefix: str
            A tag to add to the file name (prefix).
        """
        now = datetime.datetime.now()

        if prefix is not None:
            prefix = f"{prefix}_"

        plt.figure(figsize=(10, 5))
        plt.plot(hist.history['loss'], label='train')
        plt.plot(hist.history['val_loss'], label='valid')
        plt.legend()
        plt.title('Loss')
        plt.tight_layout()
        plt.savefig(f'{dir_plots}/{prefix}loss_'
                    f'{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
        if show_plots:
            plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(hist.history['csi'], label='train')
        plt.plot(hist.history['val_csi'], label='valid')
        plt.legend()
        plt.title('CSI')
        plt.tight_layout()
        plt.savefig(f'{dir_plots}/{prefix}csi_'
                    f'{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
        if show_plots:
            plt.show()
