"""
Class to handle the DL basics for models based on deep learning.
It is not meant to be used directly, but to be inherited by other classes.
"""
from .impact import Impact
from .utils.verification import compute_confusion_matrix, print_classic_scores, \
    assess_roc_auc, store_classic_scores

import os
import random
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


class ImpactDl(Impact):
    """
    The Deep Learning Impact base class.

    Parameters
    ----------
    events: Events
        The events object.
    options: ImpactDlOptions
        The model options.
    reload_trained_models: bool
        Whether to reload the previously trained models or not.
    """

    def __init__(self, events, options, reload_trained_models=False):
        super().__init__(events, options=options)
        self.reload_trained_models = reload_trained_models
        self._set_random_state()

        self.precipitation_hf = None
        self.precipitation_daily = None
        self.dem = None
        self.dg_train = None
        self.dg_val = None
        self.dg_test = None

        # Display if using GPU or CPU
        print("Built with CUDA: ", tf.test.is_built_with_cuda())
        print("Available GPU: ", tf.config.list_physical_devices('GPU'))

        # Options that will be set later
        self.factor_neg_reduction = 1

    def save_model(self, dir_output, base_name):
        """
        Save the model.

        Parameters
        ----------
        dir_output: str
            The directory where to save the model.
        base_name: str
            The base name to use for the file.
        """
        if self.model is None:
            raise ValueError("Model not defined")

        filename = f'{dir_output}/{base_name}_{self.options.run_name}.keras'
        self.model.save(filename)
        print(f"Model saved: {filename}")

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
        self._set_random_state()
        self._create_data_generator_train()
        self._create_data_generator_valid()
        self._define_model()

        # Early stopping
        callback = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True)

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
        print("Fitting the model.")
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

    def reduce_negatives_for_training(self, factor):
        """
        Reduce the number of negatives on the training set.

        Parameters
        ----------
        factor: float
            The factor to reduce the number of negatives.
        """
        self.factor_neg_reduction = factor

    def assess_model_on_all_periods(self, save_results=False, file_tag=''):
        """
        Assess the model on all periods.

        Parameters
        ----------
        save_results: bool
            Save the results to a file.
        file_tag: str
            The tag to add to the file name.
        """
        print("Creating test data generator.")
        self._create_data_generator_test()  # Implement this method in the child class

        print("Assessing the model on all periods.")
        df_res = pd.DataFrame(columns=['split'])
        df_res = self._assess_model_dg(self.dg_train, 'train', df_res)
        df_res = self._assess_model_dg(self.dg_val, 'valid', df_res)
        df_res = self._assess_model_dg(self.dg_test, 'test', df_res)

        if save_results:
            self._save_results_csv(df_res, file_tag)

    def _set_random_state(self):
        """
        Set the random state.
        """
        # Clear session and set the seed
        keras.backend.clear_session()
        if self.options.random_state is not None:
            os.environ['PYTHONHASHSEED'] = str(self.options.random_state)
            random.seed(self.options.random_state)
            np.random.seed(self.options.random_state)
            tf.random.set_seed(self.options.random_state)
            keras.utils.set_random_seed(self.options.random_state)

    def _assess_model_dg(self, dg, period_name, df_res):
        """
        Assess the model on a single period.
        """
        if self.model is None:
            raise ValueError("Model not defined")

        # Changing the batch size to speed up the evaluation
        dg.batch_size = 1024
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

        df_tmp = pd.DataFrame(columns=df_res.columns)
        df_tmp['split'] = [period_name]

        # Compute the scores
        if self.target_type == 'occurrence':
            y_pred_class = (y_pred > 0.5).astype(int)
            tp, tn, fp, fn = compute_confusion_matrix(y_obs, y_pred_class)
            print_classic_scores(tp, tn, fp, fn)
            store_classic_scores(tp, tn, fp, fn, df_tmp)
            roc = assess_roc_auc(y_obs, y_pred)
            df_tmp['ROC_AUC'] = [roc]
        else:
            rmse = np.sqrt(np.mean((y_obs - y_pred) ** 2))
            print(f"RMSE: {rmse}")
            df_tmp['RMSE'] = [rmse]
        print(f"----------------------------------------")

        df_res = pd.concat([df_res, df_tmp])

        return df_res

    def compute_f1_score_full_data(self, dg):
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
        if self.model is None:
            raise ValueError("Model not defined")

        if self.target_type != 'occurrence':
            raise ValueError("F1 score is only available for occurrence models.")

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

        # Compute the score
        y_pred_class = (y_pred > 0.5).astype(int)
        tp, tn, fp, fn = compute_confusion_matrix(y_obs, y_pred_class)
        epsilon = 1e-7  # a small constant to avoid division by zero
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
