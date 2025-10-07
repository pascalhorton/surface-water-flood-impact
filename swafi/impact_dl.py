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
    options: ImpactDlOptions
        The model options.
    events: Events
        The events object.
    reload_trained_models: bool
        Whether to reload the previously trained models or not.
    """

    def __init__(self, options, events=None, reload_trained_models=False):
        super().__init__(options, events)
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

        # Early stopping callbacks
        early_stopping_loss = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=40, restore_best_weights=True)
        early_stopping_csi = CustomEarlyStopping(
            monitor='val_csi', patience=30, min_value=0.00001)
        callbacks = [early_stopping_loss, early_stopping_csi]

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
            metrics=[CriticalSuccessIndex()]
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
            callbacks=callbacks,
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
        batch_size_orig = dg.batch_size
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

        dg.batch_size = batch_size_orig

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

        # Changing the batch size to speed up the evaluation
        batch_size_orig = dg.batch_size
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

        dg.batch_size = batch_size_orig

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
                # Ensure class weights are floats
                class_weight = {k: float(v) for k, v in self.class_weight.items()}
                loss_fn = WeightedBinaryCrossEntropy(
                    pos_weight=class_weight[1],
                    neg_weight=class_weight[0],
                    from_logits=False
                )
        else:
            loss_fn = 'mse'

        return loss_fn

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


# Define a custom early stopping callback to stop when the CSI is almost 0
class CustomEarlyStopping(keras.callbacks.Callback):
    def __init__(self, monitor='val_csi', patience=30, min_value=0.00001):
        super(CustomEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_value = min_value
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if current < self.min_value:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\nEpoch {epoch + 1}: early stopping due to {self.monitor} falling below {self.min_value} for {self.patience} consecutive epochs.")
        else:
            self.wait = 0


class WeightedBinaryCrossEntropy(keras.losses.Loss):
    """
    Serializable weighted binary cross-entropy loss.

    Supports class (sample) weighting via distinct positive / negative weights.
    Accepts labels shaped (batch,) or (batch,1) and predictions shaped (batch,), (batch,1).
    """
    def __init__(self, pos_weight=1.0, neg_weight=1.0, from_logits=False, name='weighted_binary_cross_entropy'):
        super().__init__(name=name)
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.from_logits = bool(from_logits)

    @staticmethod
    def _normalize_binary_shapes(y_true, y_pred):
        """Return y_true, y_pred squeezed to rank 1 if last dim is singleton.
        Handles common shape combos: (batch,), (batch,1)."""
        # Squeeze only if the last dim is 1
        if y_true.shape.rank == 2 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        if y_pred.shape.rank == 2 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)
        return y_true, y_pred

    def call(self, y_true, y_pred):
        # Cast to a common dtype
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        # Normalize shapes to 1-D (batch,) where possible
        y_true, y_pred = self._normalize_binary_shapes(y_true, y_pred)

        # Compute element-wise binary cross-entropy (vector of shape (batch,))
        ce = keras.metrics.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)

        # Sample weight per example based on its class
        weights_per_sample = y_true * self.pos_weight + (1.0 - y_true) * self.neg_weight
        loss = ce * weights_per_sample
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "pos_weight": self.pos_weight,
            "neg_weight": self.neg_weight,
            "from_logits": self.from_logits,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(pos_weight=config.get("pos_weight", 1.0),
                   neg_weight=config.get("neg_weight", 1.0),
                   from_logits=config.get("from_logits", False),
                   name=config.get("name", "weighted_binary_cross_entropy"))


class CriticalSuccessIndex(keras.metrics.Metric):
    """
    CSI (Critical Success Index) metric accumulating TP/FP/FN.
    Accepts predictions/labels shaped (batch,) or (batch,1).
    """
    def __init__(self, threshold=0.5, name='csi', dtype=tf.float32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name='tp', initializer='zeros', dtype=dtype)
        self.fp = self.add_weight(name='fp', initializer='zeros', dtype=dtype)
        self.fn = self.add_weight(name='fn', initializer='zeros', dtype=dtype)
        self.epsilon = tf.constant(1e-7, dtype=dtype)

    @staticmethod
    def _normalize_binary_shapes(y_true, y_pred):
        if y_true.shape.rank == 2 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        if y_pred.shape.rank == 2 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)
        return y_true, y_pred

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, self.dtype)
        y_true = tf.cast(y_true, self.dtype)
        y_true, y_pred = self._normalize_binary_shapes(y_true, y_pred)

        y_pred_bin = tf.cast(tf.greater_equal(y_pred, self.threshold), self.dtype)
        tp = tf.reduce_sum(y_true * y_pred_bin)
        fp = tf.reduce_sum((1 - y_true) * y_pred_bin)
        fn = tf.reduce_sum(y_true * (1 - y_pred_bin))

        if sample_weight is not None:
            sw = tf.cast(sample_weight, self.dtype)
            tp *= sw
            fp *= sw
            fn *= sw

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        denom = self.tp + self.fp + self.fn + self.epsilon
        return self.tp / denom

    def reset_states(self):
        self.tp.assign(0.)
        self.fp.assign(0.)
        self.fn.assign(0.)

    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold": self.threshold,
        })
        return config
