"""
Class to compute the impact function.
"""

from .impact import Impact
from .model_dl import DeepImpact
from .utils.data_generator import DataGenerator
from .utils.verification import compute_confusion_matrix, print_classic_scores, \
    assess_roc_auc, compute_score_binary

import argparse
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


class ImpactDeepLearningOptions:
    """
    The Deep Learning Impact class options.

    Attributes
    ----------
    run_id: int
        The run ID.
    target_type: str
        The target type. Options are: 'occurrence', 'damage_ratio'
    factor_neg_reduction: int
        The factor to reduce the number of negatives only for training.
    weight_denominator: int
        The weight denominator to reduce the negative class weights.
    random_state: int|None
        The random state to use for the random number generator.
        Default: 42. Set to None to not set the random seed.
    use_precip: bool
        Whether to use precipitation data (CombiPrecip) or not.
    use_dem: bool
        Whether to use DEM data or not.
    use_simple_features: bool
        Whether to use simple features (event properties and static attributes) or not.
    simple_features: list
        The list of simple features to use.
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
    transform_static: str
        The transformation to apply to the static data.
        Options are: 'standardize', 'normalize'.
    transform_2d: str
        The transformation to apply to the 2D data.
        Options are: 'standardize', 'normalize'.
    precip_trans_domain: str
        The precipitation transformation domain.
        Options are: 'domain-average', 'per-pixel'.
    batch_size: int
        The batch size.
    epochs: int
        The number of epochs.
    learning_rate: float
        The learning rate.
    dropout_rate: float
        The dropout rate.
    with_spatial_dropout: bool
        Whether to use spatial dropout or not.
    with_batchnorm: bool
        Whether to use batch normalization or not.
    nb_filters: int
        The number of filters.
    nb_conv_blocks: int
        The number of convolutional blocks.
    nb_dense_layers: int
        The number of dense layers.
    nb_dense_units: int
        The number of dense units.
    nb_dense_units_decreasing: bool
        Whether the number of dense units should decrease or not.
    inner_activation: str
        The inner activation function.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="SWAFI DL")
        self._set_parser_arguments()

        # General options
        self.run_id = 0
        self.target_type = ''
        self.factor_neg_reduction = 10
        self.weight_denominator = 5
        self.random_state = None

        # Data options
        self.use_precip = True
        self.use_dem = True
        self.use_simple_features = True
        self.simple_features = []
        self.precip_window_size = 0
        self.precip_resolution = 0
        self.precip_time_step = 0
        self.precip_days_before = 0
        self.precip_days_after = 0
        self.transform_static = ''
        self.transform_2d = ''
        self.precip_trans_domain = ''

        # Training options
        self.batch_size = 0
        self.epochs = 0
        self.learning_rate = 0

        # Model options
        self.dropout_rate = 0
        self.with_spatial_dropout = True
        self.with_batchnorm = True
        self.nb_filters = 0
        self.nb_conv_blocks = 0
        self.nb_dense_layers = 0
        self.nb_dense_units = 0
        self.nb_dense_units_decreasing = True
        self.inner_activation = ''

    def parse_args(self):
        """
        Parse the arguments.
        """
        args = self.parser.parse_args()
        self.run_id = args.run_id
        self.target_type = args.target_type
        self.factor_neg_reduction = args.factor_neg_reduction
        self.weight_denominator = args.weight_denominator
        self.random_state = args.random_state
        self.use_precip = not args.do_not_use_precip
        self.use_dem = not args.do_not_use_dem
        self.use_simple_features = not args.do_not_use_simple_features
        self.simple_features = args.simple_features
        self.precip_window_size = args.precip_window_size
        self.precip_resolution = args.precip_resolution
        self.precip_time_step = args.precip_time_step
        self.precip_days_before = args.precip_days_before
        self.precip_days_after = args.precip_days_after
        self.transform_static = args.transform_static
        self.transform_2d = args.transform_2d
        self.precip_trans_domain = args.precip_trans_domain
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.dropout_rate = args.dropout_rate
        self.with_spatial_dropout = not args.no_spatial_dropout
        self.with_batchnorm = not args.no_batchnorm
        self.nb_filters = args.nb_filters
        self.nb_conv_blocks = args.nb_conv_blocks
        self.nb_dense_layers = args.nb_dense_layers
        self.nb_dense_units = args.nb_dense_units
        self.nb_dense_units_decreasing = not args.no_dense_units_decreasing
        self.inner_activation = args.inner_activation

    def print(self):
        """
        Print the options.
        """
        print(f"Options (run {self.run_id}):")
        print("- target_type: ", self.target_type)
        print("- factor_neg_reduction: ", self.factor_neg_reduction)
        print("- weight_denominator: ", self.weight_denominator)
        print("- random_state: ", self.random_state)
        print("- use_precip: ", self.use_precip)
        print("- use_dem: ", self.use_dem)
        print("- use_simple_features: ", self.use_simple_features)
        if self.use_simple_features:
            print("- simple_features: ", self.simple_features)
        if self.use_precip:
            print("- precip_window_size: ", self.precip_window_size)
            print("- precip_resolution: ", self.precip_resolution)
            print("- precip_time_step: ", self.precip_time_step)
            print("- precip_days_before: ", self.precip_days_before)
            print("- precip_days_after: ", self.precip_days_after)
        if self.use_simple_features:
            print("- transform_static: ", self.transform_static)
        if self.use_precip:
            print("- transform_2d: ", self.transform_2d)
            print("- precip_trans_domain: ", self.precip_trans_domain)
        print("- batch_size: ", self.batch_size)
        print("- epochs: ", self.epochs)
        print("- learning_rate: ", self.learning_rate)
        print("- dropout_rate: ", self.dropout_rate)
        if self.use_precip:
            print("- with_spatial_dropout: ", self.with_spatial_dropout)
        print("- with_batchnorm: ", self.with_batchnorm)
        if self.use_precip:
            print("- nb_filters: ", self.nb_filters)
            print("- nb_conv_blocks: ", self.nb_conv_blocks)
        print("- nb_dense_layers: ", self.nb_dense_layers)
        print("- nb_dense_units: ", self.nb_dense_units)
        print("- nb_dense_units_decreasing: ", self.nb_dense_units_decreasing)
        print("- inner_activation: ", self.inner_activation)

    def is_ok(self):
        """
        Check if the options are ok.

        Returns
        -------
        bool
            Whether the options are ok or not.
        """
        # Check the precipitation parameters
        assert self.precip_window_size % self.precip_resolution == 0, \
            "precip_window_size must be divisible by precip_resolution"
        pixels_per_side = self.precip_window_size // self.precip_resolution
        assert pixels_per_side % 2 == 0, "pixels per side must be even"
        assert self.precip_days_before >= 0, "precip_days_before must be >= 0"
        assert self.precip_days_after >= 0, "precip_days_after must be >= 0"

        if not self.use_precip:
            if self.use_dem:
                self.use_dem = False
                print("Warning: DEM will not be used as precipitation is not.")

        return True

    def _set_parser_arguments(self):
        """
        Set the parser arguments.
        """
        self.parser.add_argument(
            '--run-id', type=int, default=0,
            help='The run ID')
        self.parser.add_argument(
            '--target-type', type=str, default='occurrence',
            help='The target type. Options are: occurrence, damage_ratio')
        self.parser.add_argument(
            '--factor-neg-reduction', type=int, default=10,
            help='The factor to reduce the number of negatives only for training')
        self.parser.add_argument(
            '--weight-denominator', type=int, default=5,
            help='The weight denominator to reduce the negative class weights')
        self.parser.add_argument(
            '--random-state', type=int, default=42,
            help='The random state to use for the random number generator')
        self.parser.add_argument(
            '--do-not-use-precip', action='store_true',
            help='Do not use precipitation data')
        self.parser.add_argument(
            '--do-not-use-dem', action='store_true',
            help='Do not use DEM data')
        self.parser.add_argument(
            '--do-not-use-simple-features', action='store_true',
            help='Do not use simple features (event properties and static attributes)')
        self.parser.add_argument(
            '--simple-features', nargs='+',
            default=['event', 'terrain', 'swf_map', 'flowacc', 'twi'],
            help='The list of simple features to use')
        self.parser.add_argument(
            '--precip-window-size', type=int, default=8,
            help='The precipitation window size [km]')
        self.parser.add_argument(
            '--precip-resolution', type=int, default=1,
            help='The precipitation resolution [km]')
        self.parser.add_argument(
            '--precip-time-step', type=int, default=1,
            help='The precipitation time step [h]')
        self.parser.add_argument(
            '--precip-days-before', type=int, default=8,
            help='The number of days before the event to use for the precipitation')
        self.parser.add_argument(
            '--precip-days-after', type=int, default=3,
            help='The number of days after the event to use for the precipitation')
        self.parser.add_argument(
            '--transform-static', type=str, default='standardize',
            help='The transformation to apply to the static data')
        self.parser.add_argument(
            '--transform-2d', type=str, default='standardize',
            help='The transformation to apply to the 2D data')
        self.parser.add_argument(
            '--precip-trans-domain', type=str, default='per-pixel',
            help='The precipitation transformation domain. '
                 'Options are: domain-average, per-pixel')
        self.parser.add_argument(
            '--batch-size', type=int, default=32,
            help='The batch size')
        self.parser.add_argument(
            '--epochs', type=int, default=100,
            help='The number of epochs')
        self.parser.add_argument(
            '--learning-rate', type=float, default=0.001,
            help='The learning rate')
        self.parser.add_argument(
            '--dropout-rate', type=float, default=0.2,
            help='The dropout rate')
        self.parser.add_argument(
            '--no-spatial-dropout', action='store_true',
            help='Do not use spatial dropout')
        self.parser.add_argument(
            '--no-batchnorm', action='store_true',
            help='Do not use batch normalization')
        self.parser.add_argument(
            '--nb-filters', type=int, default=64,
            help='The number of filters')
        self.parser.add_argument(
            '--nb-conv-blocks', type=int, default=3,
            help='The number of convolutional blocks')
        self.parser.add_argument(
            '--nb-dense-layers', type=int, default=4,
            help='The number of dense layers')
        self.parser.add_argument(
            '--nb-dense-units', type=int, default=256,
            help='The number of dense units')
        self.parser.add_argument(
            '--no-dense-units-decreasing', action='store_true',
            help='The number of dense units should not decrease')
        self.parser.add_argument(
            '--inner-activation', type=str, default='relu',
            help='The inner activation function')


class ImpactDeepLearning(Impact):
    """
    The Deep Learning Impact class.

    Parameters
    ----------
    events: Events
        The events object.
    options: ImpactDeepLearningOptions
        The model options.
    reload_trained_models: bool
        Whether to reload the previously trained models or not.
    """

    def __init__(self, events, options, reload_trained_models=False):
        super().__init__(events, target_type=options.target_type,
                         random_state=options.random_state)
        self.options = options
        self.reload_trained_models = reload_trained_models

        self.precipitation = None
        self.dem = None
        self.dg_train = None
        self.dg_val = None
        self.dg_test = None

        if not self.options.is_ok():
            raise ValueError("Options are not ok.")

        # Display if using GPU or CPU
        print("Built with CUDA: ", tf.test.is_built_with_cuda())
        print("Available GPU: ", tf.config.list_physical_devices('GPU'))

        # Options that will be set later
        self.factor_neg_reduction = 1

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
        pixels_per_side = (self.options.precip_window_size //
                           self.options.precip_resolution)
        if self.options.use_precip:
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
        keras.utils.set_random_seed(self.options.random_state)

        # Define the optimizer
        optimizer = self._define_optimizer(
            n_samples=len(self.dg_train),
            lr_method='constant',
            lr=self.options.learning_rate)

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
            epochs=self.options.epochs,
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
        print("Assessing the model on all periods.")
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
            batch_size=self.options.batch_size,
            shuffle=True,
            precip_window_size=self.options.precip_window_size,
            precip_resolution=self.options.precip_resolution,
            precip_time_step=self.options.precip_time_step,
            precip_days_before=self.options.precip_days_before,
            precip_days_after=self.options.precip_days_after,
            tmp_dir=self.tmp_dir,
            transform_static=self.options.transform_static,
            transform_2d=self.options.transform_2d,
            precip_transformation_domain=self.options.precip_trans_domain,
            log_transform_precip=True,
            debug=True
        )

        if self.factor_neg_reduction != 1:
            self.dg_train.reduce_negatives(self.factor_neg_reduction)

        if self.options.use_precip:
            self.dg_train.prepare_precip_data()

    def _create_data_generator_valid(self):
        self.dg_val = DataGenerator(
            event_props=self.events_valid,
            x_static=self.x_valid,
            x_precip=self.precipitation,
            x_dem=self.dem,
            y=self.y_valid,
            batch_size=self.options.batch_size,
            shuffle=True,
            precip_window_size=self.options.precip_window_size,
            precip_resolution=self.options.precip_resolution,
            precip_time_step=self.options.precip_time_step,
            precip_days_before=self.options.precip_days_before,
            precip_days_after=self.options.precip_days_after,
            tmp_dir=self.tmp_dir,
            transform_static=self.options.transform_static,
            transform_2d=self.options.transform_2d,
            precip_transformation_domain=self.options.precip_trans_domain,
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

        if self.options.use_precip:
            self.dg_val.prepare_precip_data()

    def _create_data_generator_test(self):
        self.dg_test = DataGenerator(
            event_props=self.events_test,
            x_static=self.x_test,
            x_precip=self.precipitation,
            x_dem=self.dem,
            y=self.y_test,
            batch_size=self.options.batch_size,
            shuffle=True,
            precip_window_size=self.options.precip_window_size,
            precip_resolution=self.options.precip_resolution,
            precip_time_step=self.options.precip_time_step,
            precip_days_before=self.options.precip_days_before,
            precip_days_after=self.options.precip_days_after,
            tmp_dir=self.tmp_dir,
            transform_static=self.options.transform_static,
            transform_2d=self.options.transform_2d,
            precip_transformation_domain=self.options.precip_trans_domain,
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

        if self.options.use_precip:
            self.dg_test.prepare_precip_data()

    def _define_model(self, input_2d_size, input_1d_size):
        """
        Define the model.
        """
        self.model = DeepImpact(
            task=self.target_type,
            options=self.options,
            input_2d_size=input_2d_size,
            input_1d_size=input_1d_size,
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
            decay_steps = self.options.epochs * (n_samples / self.options.batch_size)
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
                pickle.dumps(self.df.shape) +
                pickle.dumps(self.df.columns) +
                pickle.dumps(self.df.iloc[0]) +
                pickle.dumps(self.features) +
                pickle.dumps(self.class_weight) +
                pickle.dumps(self.options.random_state) +
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
        if not self.options.use_precip:
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
                if self.options.precip_resolution != 1:
                    precipitation = precipitation.coarsen(
                        x=self.options.precip_resolution,
                        y=self.options.precip_resolution,
                        boundary='trim'
                    ).mean()

            # Aggregate the precipitation at the desired time step
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                if self.options.precip_time_step != 1:
                    precipitation = precipitation.resample(
                        time=f'{self.options.precip_time_step}h',
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
        if not self.options.use_precip:
            print("DEM is not used and is therefore not loaded.")
            return

        assert dem.ndim == 2, "DEM must be 2D"

        # Adapt the spatial resolution
        if self.options.precip_resolution != 1:
            dem = dem.coarsen(
                x=self.options.precip_resolution,
                y=self.options.precip_resolution,
                boundary='trim'
            ).mean()

        if self.precipitation is not None:
            assert dem.shape == self.precipitation.isel(time=0).shape, \
                "DEM and precipitation must have the same shape"
        self.dem = dem

    def _compute_hash_precip_full_data(self, precipitation):
        tag_data = (
                pickle.dumps(self.options.precip_resolution) +
                pickle.dumps(self.options.precip_time_step) +
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
