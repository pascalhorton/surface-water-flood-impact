"""
Class to compute the impact function with the CNN model.
"""

from .impact import Impact
from .impact_cnn_options import ImpactCnnOptions
from .impact_cnn_model import ModelCnn
from .utils.data_generator import DataGenerator
from .utils.verification import compute_confusion_matrix, print_classic_scores, \
    assess_roc_auc, compute_score_binary

import copy
import hashlib
import pickle
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime

has_optuna = False
try:
    import optuna
    has_optuna = True
except ImportError:
    pass

DEBUG = False


class ImpactCnn(Impact):
    """
    The CNN Deep Learning Impact class.

    Parameters
    ----------
    events: Events
        The events object.
    options: ImpactCnnOptions
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

    def save_model(self, dir_output):
        """
        Save the model.

        Parameters
        ----------
        dir_output: str
            The directory where to save the model.
        """
        if self.model is None:
            raise ValueError("Model not defined")

        filename = f'{dir_output}/model_cnn_{self.options.run_name}.keras'
        self.model.save(filename)
        print(f"Model saved: {filename}")

    def copy(self):
        """
        Make a copy of the object.
        Returns
        -------
        ImpactCnn
            The copy of the object.
        """
        return copy.deepcopy(self)

    def fit(self, tag=None, do_plot=True, dir_plots=None, show_plots=False,
            silent=False):
        """
        Fit the model.

        Parameters
        ----------
        tag: str
            A tag to add to the file name.
        do_plot: bool
            Whether to plot the training history or not.
        dir_plots: str
            The directory where to save the plots.
        show_plots: bool
            Whether to show the plots or not.
        silent: bool
            Hide model summary and training progress.
        """
        self._create_data_generator_train()
        self._create_data_generator_valid()

        # Define the model
        pixels_per_side = (self.options.precip_window_size //
                           self.options.precip_resolution)
        if self.options.use_precip:
            self._define_model(input_3d_size=[pixels_per_side,
                                              pixels_per_side,
                                              self.dg_train.get_third_dim_size(),
                                              1],  # 1 channel
                               input_1d_size=self.x_train.shape[1:])
        else:
            self._define_model(input_3d_size=None,
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
        if not silent:
            self.model.model.summary()

        # Fit the model
        verbose = 1 if show_plots else 2
        verbose = 0 if silent else verbose
        hist = self.model.fit(
            self.dg_train,
            epochs=self.options.epochs,
            validation_data=self.dg_val,
            callbacks=[callback],
            verbose=verbose,
            shuffle=False
        )

        # Plot the training history
        if do_plot:
            self._plot_training_history(hist, dir_plots, show_plots, tag)

    def optimize_model_with_optuna(self, n_trials=100, n_jobs=4, dir_plots=None):
        """
        Optimize the model with Optuna.

        Parameters
        ----------
        n_trials: int
            The number of trials.
        n_jobs: int
            The number of jobs to run in parallel.
        dir_plots: str
            The directory where to save the plots.
        """
        if not has_optuna:
            raise ValueError("Optuna is not installed")

        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=n_trials, n_jobs=n_jobs)

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Fit the model with the best parameters
        if not self.options.generate_for_optuna(trial):
            print("The parameters are not valid.")
            return float('-inf')

        self.compute_balanced_class_weights()
        self.compute_corrected_class_weights(
            weight_denominator=self.options.weight_denominator)
        self.fit(dir_plots=dir_plots, tag='best_optuna_' + self.options.run_name)

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

    def compute_f1_score(self, dg):
        """
        Compute the F1 score on the given set.

        Parameters
        ----------
        dg: DataGenerator
            The data generator.

        Returns
        -------
        float
            The F1 score.
        """
        n_batches = dg.__len__()
        epsilon = 1e-7  # a small constant to avoid division by zero

        # Predict
        all_pred = []
        all_obs = []
        for i in range(n_batches):
            x, y = dg.__getitem__(i)
            all_obs.append(y)
            y_pred_batch = self.model.predict(x, verbose=0)

            # Get rid of the single dimension
            y_pred_batch = y_pred_batch.squeeze()
            all_pred.append(y_pred_batch)

        # Concatenate predictions and obs from all batches
        y_pred = np.concatenate(all_pred, axis=0)
        y_obs = np.concatenate(all_obs, axis=0)

        y_pred_class = (y_pred > 0.5).astype(int)
        tp, tn, fp, fn = compute_confusion_matrix(y_obs, y_pred_class)
        f1 = 2 * tp / (2 * tp + fp + fn + epsilon)

        return f1

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
        epsilon = 1e-7  # a small constant to avoid division by zero
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
            transform_precip=self.options.transform_precip,
            log_transform_precip=self.options.log_transform_precip,
            debug=DEBUG
        )

        if (self.options.use_precip and self.precipitation is not None and
                self.options.precip_window_size / self.options.precip_resolution == 1):
            print("Preloading all precipitation data.")
            all_cids = self.df['cid'].unique()
            self.precipitation.preload_all_cid_data(all_cids)

        if self.factor_neg_reduction != 1:
            self.dg_train.reduce_negatives(self.factor_neg_reduction)

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
            transform_precip=self.options.transform_precip,
            log_transform_precip=self.options.log_transform_precip,
            mean_static=self.dg_train.mean_static,
            std_static=self.dg_train.std_static,
            mean_precip=self.dg_train.mean_precip,
            std_precip=self.dg_train.std_precip,
            min_static=self.dg_train.min_static,
            max_static=self.dg_train.max_static,
            q99_precip=self.dg_train.q99_precip,
            debug=DEBUG
        )

        if self.factor_neg_reduction != 1:
            self.dg_val.reduce_negatives(self.factor_neg_reduction)

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
            transform_precip=self.options.transform_precip,
            log_transform_precip=self.options.log_transform_precip,
            mean_static=self.dg_train.mean_static,
            std_static=self.dg_train.std_static,
            mean_precip=self.dg_train.mean_precip,
            std_precip=self.dg_train.std_precip,
            min_static=self.dg_train.min_static,
            max_static=self.dg_train.max_static,
            q99_precip=self.dg_train.q99_precip,
            debug=DEBUG
        )

    def _define_model(self, input_3d_size, input_1d_size):
        """
        Define the model.
        """
        self.model = ModelCnn(
            task=self.target_type,
            options=self.options,
            input_3d_size=input_3d_size,
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
        model_hashed_name = f'cnn_model_{hashlib.md5(tag_model).hexdigest()}.pickle'
        tmp_filename = self.tmp_dir / model_hashed_name

        return tmp_filename

    def set_precipitation(self, precipitation):
        """
        Set the precipitation data.

        Parameters
        ----------
        precipitation: Precipitation|None
            The precipitation data.
        """
        if precipitation is None:
            return

        if not self.options.use_precip:
            print("Precipitation is not used and is therefore not loaded.")
            return

        precipitation.prepare_data(resolution=self.options.precip_resolution,
                                   time_step=self.options.precip_time_step)

        # Check the shape of the precipitation and the DEM
        if self.dem is not None:
            # Select the same domain as the DEM
            precipitation.generate_pickles_for_subdomain(self.dem.x, self.dem.y)

        self.precipitation = precipitation

    def set_dem(self, dem):
        """
        Set the DEM data.

        Parameters
        ----------
        dem: xarray.Dataset|None
            The DEM data.
        """
        if dem is None:
            return

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

        self.dem = dem

    def reduce_spatial_domain(self, precip_window_size):
        """
        Restrict the spatial domain of the precipitation and DEM data.

        Parameters
        ----------
        precip_window_size: int
            The precipitation window size [km].
        """
        precip_window_size_m = 15 * 1000
        if precip_window_size > 15:
            precip_window_size_m = precip_window_size * 1000
        x_min = self.df['x'].min() - precip_window_size_m / 2
        x_max = self.df['x'].max() + precip_window_size_m / 2
        y_min = self.df['y'].min() - precip_window_size_m / 2
        y_max = self.df['y'].max() + precip_window_size_m / 2
        if self.precipitation is not None:
            x_axis = self.precipitation.get_x_axis_for_bounds(x_min, x_max)
            y_axis = self.precipitation.get_y_axis_for_bounds(y_min, y_max)
            self.precipitation.generate_pickles_for_subdomain(x_axis, y_axis)
        if self.dem is not None:
            self.dem = self.dem.sel(
                x=slice(x_min, x_max),
                y=slice(y_max, y_min)
            )

    def remove_events_without_precipitation_data(self):
        """
        Remove the events at the period limits.
        """
        if self.precipitation is None:
            return

        # Extract events dates
        events = self.df[['e_end', 'date_claim']].copy()
        events.rename(columns={'date_claim': 'date'}, inplace=True)
        events['e_end'] = pd.to_datetime(events['e_end']).dt.date
        events['date'] = pd.to_datetime(events['date']).dt.date

        # Fill NaN values with the event end date (as date, not datetime)
        events['date'] = events['date'].fillna(events['e_end'])

        # Precipitation period
        p_start = pd.to_datetime(f'{self.precipitation.year_start}-01-01').date()
        p_end = pd.to_datetime(f'{self.precipitation.year_end}-12-31').date()

        if self.options.precip_days_before > 0:
            self.df = self.df[events['date'] > p_start + pd.Timedelta(
                days=self.options.precip_days_before)]
            events = events[events['date'] > p_start + pd.Timedelta(
                days=self.options.precip_days_before)]
        if self.options.precip_days_after > 0:
            self.df = self.df[events['date'] < p_end - pd.Timedelta(
                days=self.options.precip_days_after)]

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
