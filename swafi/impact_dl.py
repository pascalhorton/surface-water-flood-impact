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
    optimize_decision_threshold: bool
        Whether to optimize the decision threshold from validation data or not.
    """

    def __init__(self, options, events=None, reload_trained_models=False, optimize_decision_threshold=False):
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

        # Decision threshold for classification; tuned from validation by default
        self.optimize_decision_threshold = optimize_decision_threshold
        self.decision_threshold = 0.5

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
        early_stopping_csi = keras.callbacks.EarlyStopping(
            monitor='val_csi', patience=40, verbose=1,
            restore_best_weights=True, mode='max')
        # Fallback: stop if CSI drops to near-zero and stays there
        early_stopping_no_skill = CustomEarlyStopping(
            monitor='val_csi', patience=30, min_value=0.00001)
        callbacks = [early_stopping_csi, early_stopping_no_skill]

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
            metrics=[CriticalSuccessIndex(), F1Score()],
            run_eagerly=DEBUG  # Set to True for debugging purposes
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

        # Determine a good decision threshold from validation data if it's a classifier
        if self.target_type == 'occurrence' and self.options.loss_function == 'bce' and should_optimize_threshold and self.dg_val is not None:
            thr, metric_name, metric_value = self._find_optimal_threshold(self.dg_val, metric='f1')
            if thr is not None:
                self.decision_threshold = float(thr)
                print(f"Selected decision threshold from validation ({metric_name}): {self.decision_threshold:.4f} (score={metric_value:.4f})")
            else:
                print("Could not determine an optimal threshold from validation; using default 0.5")
                self.decision_threshold = 0.5

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
            # Ensure observations are 1D arrays to avoid broadcasting issues
            all_obs.append(np.asarray(y).squeeze())
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
            thr = self.decision_threshold
            print(f"Using decision threshold: {thr:.4f}")
            y_pred_class = (y_pred >= thr).astype(int)
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
            # Ensure observations are 1D arrays to avoid broadcasting issues
            all_obs.append(np.asarray(y).squeeze())
            y_pred_batch = self.model.predict(x, verbose=0)

            # Get rid of the single dimension
            y_pred_batch = y_pred_batch.squeeze()
            all_pred.append(y_pred_batch)

        dg.batch_size = batch_size_orig

        # Concatenate predictions and obs from all batches
        y_pred = np.concatenate(all_pred, axis=0)
        y_obs = np.concatenate(all_obs, axis=0)

        # Compute the score
        thr = self.decision_threshold
        y_pred_class = (y_pred >= thr).astype(int)
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
            # Ensure class weights are floats
            class_weight = {k: float(v) for k, v in self.class_weight.items()}

            # Get loss type from options if available
            loss_type = getattr(self.options, 'loss_function', 'bce')

            if loss_type == 'soft_f1':
                # Use Soft F1 Loss - directly optimizes F1 score
                # class_weight[1] handles positive class importance
                loss_fn = SoftF1Loss(
                    beta=1.0,
                    class_weight=class_weight[1],
                    from_logits=False
                )
                print(f"Using Soft F1 Loss (class_weight={class_weight[1]:.2f})")
            elif loss_type == 'soft_csi':
                # Use Soft CSI Loss - directly optimizes CSI
                loss_fn = SoftCSILoss(
                    class_weight=class_weight[1],
                    from_logits=False
                )
                print(f"Using Soft CSI Loss (class_weight={class_weight[1]:.2f})")
            else:  # 'bce' or default
                # Use weighted binary cross-entropy (original)
                loss_fn = WeightedBinaryCrossEntropy(
                    pos_weight=class_weight[1],
                    neg_weight=class_weight[0],
                    from_logits=False
                )
                print(f"Using Weighted BCE (pos_weight={class_weight[1]:.2f}, neg_weight={class_weight[0]:.2f})")
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

    def _find_optimal_threshold(self, dg, metric='f1', thresholds=None):
        """
        Compute predicted probabilities on the full dataset of the given generator
        and select the threshold that maximizes the chosen metric on that set.

        Parameters
        ----------
        dg: DataGenerator
            The data generator to evaluate (usually validation).
        metric: str
            'f1' or 'csi' to choose which metric to maximize.
        thresholds: array-like or None
            Optional set of thresholds to evaluate. If None, uses np.linspace(0,1,201).

        Returns
        -------
        (best_thr, metric_name, best_score)
            best_thr is None if it couldn't be determined (e.g., no positives).
        """
        if self.model is None:
            return None, metric, np.nan
        if getattr(self, 'target_type', 'occurrence') != 'occurrence':
            return None, metric, np.nan

        # Predict on full dataset
        batch_size_orig = dg.batch_size
        dg.batch_size = 1024
        n_batches = dg.get_number_of_batches_for_full_dataset()
        all_pred, all_obs = [], []
        for i in range(n_batches):
            x, y = dg.get_ordered_batch_from_full_dataset(i)
            all_obs.append(np.asarray(y).squeeze())
            y_pred_batch = self.model.predict(x, verbose=0).squeeze()
            all_pred.append(y_pred_batch)
        dg.batch_size = batch_size_orig

        y_pred = np.concatenate(all_pred, axis=0)
        y_obs = np.concatenate(all_obs, axis=0).astype(int)

        # Edge cases
        n_pos = int(np.sum(y_obs))
        n_neg = int(len(y_obs) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return None, metric, np.nan

        if thresholds is None:
            thresholds = np.linspace(0.0, 1.0, 201)

        best_thr = None
        best_score = -np.inf
        eps = 1e-7
        # Initialize metric_name based on requested metric
        metric_name = 'CSI' if metric.lower() == 'csi' else 'F1'
        for thr in thresholds:
            y_cls = (y_pred >= thr).astype(int)
            tp = int(np.sum((y_obs == 1) & (y_cls == 1)))
            fp = int(np.sum((y_obs == 0) & (y_cls == 1)))
            fn = int(np.sum((y_obs == 1) & (y_cls == 0)))
            if metric.lower() == 'csi':
                score = tp / (tp + fp + fn + eps)
            else:  # F1 by default
                score = 2 * tp / (2 * tp + fp + fn + eps)
            if score > best_score:
                best_score = score
                best_thr = thr

        return best_thr, metric_name, float(best_score)


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

    Parameters
    ----------
    pos_weight : float
        Multiplicative weight applied to positive (y=1) examples.
    neg_weight : float
        Multiplicative weight applied to negative (y=0) examples.
    from_logits : bool
        If True, y_pred is treated as logits; otherwise probabilities.
    normalize : bool
        If True, loss is sum(weight * BCE) / sum(weights) (keeps magnitude
        comparable to unweighted BCE). If False, it's mean(weight * BCE), which
        scales with average weight and can inflate reported loss.
    """
    def __init__(self, pos_weight=1.0, neg_weight=1.0, from_logits=False,
                 normalize=False, name='weighted_binary_cross_entropy'):
        super().__init__(name=name)
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.from_logits = bool(from_logits)
        self.normalize = bool(normalize)

    @staticmethod
    def _expand_shapes(y_true, y_pred):
        if y_true.shape.rank == 1:
            y_true = tf.expand_dims(y_true, axis=-1)
        if y_pred.shape.rank == 1:
            y_pred = tf.expand_dims(y_pred, axis=-1)
        return y_true, y_pred

    def call(self, y_true, y_pred):
        y_true, y_pred = self._expand_shapes(y_true, y_pred)

        ce = keras.metrics.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)

        weights = y_true * self.pos_weight + (1.0 - y_true) * self.neg_weight
        weighted = ce * weights
        if self.normalize:
            loss = tf.reduce_sum(weighted) / (tf.reduce_sum(weights) + 1e-7)
        else:
            loss = tf.reduce_mean(weighted)

        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "pos_weight": self.pos_weight,
            "neg_weight": self.neg_weight,
            "from_logits": self.from_logits,
            "normalize": self.normalize
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(pos_weight=config.get("pos_weight", 1.0),
                   neg_weight=config.get("neg_weight", 1.0),
                   from_logits=config.get("from_logits", False),
                   normalize=config.get("normalize", True),
                   name=config.get("name", "weighted_binary_cross_entropy"))


class SoftF1Loss(keras.losses.Loss):
    """
    Differentiable soft F1 loss (equivalent to soft CSI with beta=1).

    Directly optimizes F1/CSI by computing soft TP/FP/FN from probabilities
    instead of hard predictions. Uses y_pred as soft predictions (no thresholding).

    Loss = 1 - F1_score where F1 = 2*TP / (2*TP + FP + FN)
    CSI = TP / (TP + FP + FN) is similar but slightly different weighting.

    Parameters
    ----------
    beta : float
        Beta parameter for F-beta score. Use beta=1 for F1 (default).
        Use beta → ∞ to approximate CSI behavior.
    class_weight : float
        Weight multiplier for positive class to handle imbalance.
        Effectively scales TP and FN by this factor.
    smooth : float
        Smoothing epsilon to avoid division by zero.
    from_logits : bool
        If True, apply sigmoid to y_pred first.
    """
    def __init__(self, beta=1.0, class_weight=1.0, smooth=1e-7,
                 from_logits=False, name='soft_f1_loss'):
        super().__init__(name=name)
        self.beta = float(beta)
        self.beta_squared = self.beta ** 2
        self.class_weight = float(class_weight)
        self.smooth = float(smooth)
        self.from_logits = bool(from_logits)

    def call(self, y_true, y_pred):
        # Ensure correct shapes
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if y_true.shape.rank == 1:
            y_true = tf.expand_dims(y_true, axis=-1)
        if y_pred.shape.rank == 1:
            y_pred = tf.expand_dims(y_pred, axis=-1)

        # Apply sigmoid if needed
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        # Soft confusion matrix components (using probabilities directly)
        # TP: when both true and predicted are high
        tp = tf.reduce_sum(y_true * y_pred * self.class_weight)

        # FP: when true is low but predicted is high
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)

        # FN: when true is high but predicted is low
        fn = tf.reduce_sum(y_true * (1.0 - y_pred) * self.class_weight)

        # Soft F-beta score
        numerator = (1.0 + self.beta_squared) * tp + self.smooth
        denominator = (1.0 + self.beta_squared) * tp + self.beta_squared * fn + fp + self.smooth

        soft_f_beta = numerator / denominator

        # Return loss (1 - F-beta to minimize)
        return 1.0 - soft_f_beta

    def get_config(self):
        config = super().get_config()
        config.update({
            "beta": self.beta,
            "class_weight": self.class_weight,
            "smooth": self.smooth,
            "from_logits": self.from_logits,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            beta=config.get("beta", 1.0),
            class_weight=config.get("class_weight", 1.0),
            smooth=config.get("smooth", 1e-7),
            from_logits=config.get("from_logits", False),
            name=config.get("name", "soft_f1_loss")
        )


class SoftCSILoss(keras.losses.Loss):
    """
    Differentiable soft CSI (Critical Success Index) loss.

    CSI = TP / (TP + FP + FN)

    This is mathematically similar to F1 but weights TP/FP/FN differently.
    CSI tends to be more sensitive to false alarms than F1.

    Parameters
    ----------
    class_weight : float
        Weight multiplier for positive class to handle imbalance.
    smooth : float
        Smoothing epsilon to avoid division by zero.
    from_logits : bool
        If True, apply sigmoid to y_pred first.
    """
    def __init__(self, class_weight=1.0, smooth=1e-7,
                 from_logits=False, name='soft_csi_loss'):
        super().__init__(name=name)
        self.class_weight = float(class_weight)
        self.smooth = float(smooth)
        self.from_logits = bool(from_logits)

    def call(self, y_true, y_pred):
        # Ensure correct shapes
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if y_true.shape.rank == 1:
            y_true = tf.expand_dims(y_true, axis=-1)
        if y_pred.shape.rank == 1:
            y_pred = tf.expand_dims(y_pred, axis=-1)

        # Apply sigmoid if needed
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        # Soft confusion matrix components
        tp = tf.reduce_sum(y_true * y_pred * self.class_weight)
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1.0 - y_pred) * self.class_weight)

        # Soft CSI score
        soft_csi = (tp + self.smooth) / (tp + fp + fn + self.smooth)

        # Return loss (1 - CSI to minimize)
        return 1.0 - soft_csi

    def get_config(self):
        config = super().get_config()
        config.update({
            "class_weight": self.class_weight,
            "smooth": self.smooth,
            "from_logits": self.from_logits,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            class_weight=config.get("class_weight", 1.0),
            smooth=config.get("smooth", 1e-7),
            from_logits=config.get("from_logits", False),
            name=config.get("name", "soft_csi_loss")
        )


class CriticalSuccessIndex(keras.metrics.Metric):
    """
    CSI (Critical Success Index) metric accumulating TP/FP/FN.
    """
    def __init__(self, threshold=0.5, name='csi', dtype=tf.float32):
        super().__init__(name=name)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name='tp', shape=(), initializer='zeros', dtype=dtype)
        self.fp = self.add_weight(name='fp', shape=(), initializer='zeros', dtype=dtype)
        self.fn = self.add_weight(name='fn', shape=(), initializer='zeros', dtype=dtype)
        self.epsilon = tf.constant(1e-7, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure tensors and dynamic-safe squeezing of last dim when it's 1
        y_pred = tf.cast(y_pred, self.dtype)
        y_true = tf.cast(y_true, self.dtype)

        def _maybe_squeeze(a):
            a = tf.convert_to_tensor(a)
            rank = tf.rank(a)
            last_dim = tf.shape(a)[-1]
            return tf.cond(tf.logical_and(tf.equal(rank, 2), tf.equal(last_dim, 1)),
                           lambda: tf.squeeze(a, axis=-1),
                           lambda: a)

        y_true = _maybe_squeeze(y_true)
        y_pred = _maybe_squeeze(y_pred)

        y_pred_bin = tf.cast(tf.greater_equal(y_pred, self.threshold), self.dtype)

        if sample_weight is not None:
            sw = tf.cast(sample_weight, self.dtype)
            # Ensure sample_weight is broadcastable to batch shape
            tp = tf.reduce_sum(y_true * y_pred_bin * sw)
            fp = tf.reduce_sum((1 - y_true) * y_pred_bin * sw)
            fn = tf.reduce_sum(y_true * (1 - y_pred_bin) * sw)
        else:
            tp = tf.reduce_sum(y_true * y_pred_bin)
            fp = tf.reduce_sum((1 - y_true) * y_pred_bin)
            fn = tf.reduce_sum(y_true * (1 - y_pred_bin))

        # Update state variables
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


class F1Score(keras.metrics.Metric):
    """
    F1 Score metric accumulating TP/FP/FN.
    """
    def __init__(self, threshold=0.5, name='f1_score', dtype=tf.float32):
        super().__init__(name=name)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name='tp', shape=(), initializer='zeros', dtype=dtype)
        self.fp = self.add_weight(name='fp', shape=(), initializer='zeros', dtype=dtype)
        self.fn = self.add_weight(name='fn', shape=(), initializer='zeros', dtype=dtype)
        self.epsilon = tf.constant(1e-7, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Dynamic-safe squeezing like in CSI
        y_pred = tf.cast(y_pred, self.dtype)
        y_true = tf.cast(y_true, self.dtype)

        def _maybe_squeeze(a):
            a = tf.convert_to_tensor(a)
            rank = tf.rank(a)
            last_dim = tf.shape(a)[-1]
            return tf.cond(tf.logical_and(tf.equal(rank, 2), tf.equal(last_dim, 1)),
                           lambda: tf.squeeze(a, axis=-1),
                           lambda: a)

        y_true = _maybe_squeeze(y_true)
        y_pred = _maybe_squeeze(y_pred)

        y_pred_bin = tf.cast(tf.greater_equal(y_pred, self.threshold), self.dtype)

        if sample_weight is not None:
            sw = tf.cast(sample_weight, self.dtype)
            tp = tf.reduce_sum(y_true * y_pred_bin * sw)
            fp = tf.reduce_sum((1 - y_true) * y_pred_bin * sw)
            fn = tf.reduce_sum(y_true * (1 - y_pred_bin) * sw)
        else:
            tp = tf.reduce_sum(y_true * y_pred_bin)
            fp = tf.reduce_sum((1 - y_true) * y_pred_bin)
            fn = tf.reduce_sum(y_true * (1 - y_pred_bin))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + self.epsilon)
        recall = self.tp / (self.tp + self.fn + self.epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + self.epsilon)
        return f1_score

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
