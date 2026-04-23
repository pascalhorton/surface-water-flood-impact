"""
Class to handle the DL basics for models based on deep learning.
It is not meant to be used directly, but to be inherited by other classes.
"""
from .impact import Impact
from .utils.verification import compute_confusion_matrix, print_classic_scores, \
    assess_roc_auc, store_classic_scores

import logging
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

logger = logging.getLogger(__name__)


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
        logger.info("Built with CUDA:  %s", tf.test.is_built_with_cuda())
        logger.info("Available GPU:  %s", tf.config.list_physical_devices('GPU'))

        # Options that will be set later
        self.factor_neg_reduction = 1

        # Decision threshold for classification; tuned from validation by default
        self.optimize_decision_threshold = optimize_decision_threshold
        self.decision_threshold = 0.5

    def save_model(self, dir_output, base_name='model'):
        """
        Save the model.

        Parameters
        ----------
        dir_output: str
            The directory where to save the model.
        base_name: str
            The base name to use for the file. The run name will be appended.
            Default is 'model'.
        """
        if self.model is None:
            raise ValueError("Model not defined")

        filename = f'{dir_output}/{base_name}_{self.options.run_name}.keras'
        self.model.save(filename)
        logger.info("Model saved: %s", filename)

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
            lr_method=self.options.lr_method,
            lr=self.options.learning_rate,
            init_lr=self.options.learning_rate)

        # Get loss function
        loss_fn = self._get_loss_function()

        # Create instances of ROC-AUC and PR-AUC metrics to track during training
        roc_auc = keras.metrics.AUC(name='ROC_AUC', curve='ROC')
        pr_auc = keras.metrics.AUC(name='PR_AUC', curve='PR')

        # Compile the model
        self.model.compile(
            loss=loss_fn,
            optimizer=optimizer,
            metrics=[CriticalSuccessIndex(), F1Score(), roc_auc, pr_auc],
            run_eagerly=DEBUG,  # Set to True for debugging purposes
            steps_per_execution=self.options.steps_per_execution
        )

        # Print the model summary
        if not silent:
            self.model.model.summary()

        if self.options.disable_xla_autotune:
            tf.config.optimizer.set_jit(False)
            logger.info("XLA JIT disabled to avoid cuDNN fused-kernel autotuner failures")

        # Fit the model
        logger.info("Fitting the model.")
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
        logger.info("Creating test data generator.")
        self._create_data_generator_test()  # Implement this method in the child class

        # Determine a good decision threshold from validation data if it's a classifier
        if self.target_type == 'occurrence' and self.dg_val is not None:
            thr, metric_name, metric_value = self._find_optimal_threshold(self.dg_val, metric='f1')
            if thr is not None:
                self.decision_threshold = float(thr)
                logger.info("Selected decision threshold from validation (%s): %.4f (score=%.4f)",
                            metric_name, self.decision_threshold, metric_value)
            else:
                logger.warning("Could not determine an optimal threshold from validation; using default 0.5")
                self.decision_threshold = 0.5

        logger.info("Assessing the model on all periods.")
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

        logger.info("\nSplit: %s", period_name)

        df_tmp = pd.DataFrame(columns=df_res.columns)
        df_tmp['split'] = [period_name]

        # Compute the scores
        if self.target_type == 'occurrence':
            thr = self.decision_threshold
            logger.info("Using decision threshold: %.4f", thr)
            y_pred_class = (y_pred >= thr).astype(int)
            tp, tn, fp, fn = compute_confusion_matrix(y_obs, y_pred_class)
            print_classic_scores(tp, tn, fp, fn)
            store_classic_scores(tp, tn, fp, fn, df_tmp)
            roc = assess_roc_auc(y_obs, y_pred)
            df_tmp['ROC_AUC'] = [roc]
        else:
            rmse = np.sqrt(np.mean((y_obs - y_pred) ** 2))
            logger.info("RMSE: %s", rmse)
            df_tmp['RMSE'] = [rmse]
        logger.info("----------------------------------------")

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
            logger.info("Class weights: %s", class_weight)

            # Get loss type from options if available
            loss_type = getattr(self.options, 'loss_function', 'focal')

            if loss_type == 'wbce':  # weighted binary cross-entropy
                loss_fn = WeightedBinaryCrossEntropy(
                    pos_weight=class_weight[1],
                    neg_weight=class_weight[0],
                    from_logits=False
                )
                logger.info("Using Weighted BCE (pos_weight=%.2f, neg_weight=%.2f)",
                            class_weight[1], class_weight[0])

            elif loss_type == 'focal':  # focal loss
                # Convert pos_weight to alpha for focal loss
                pos_weight = class_weight[1]
                alpha = pos_weight / (1.0 + pos_weight)

                loss_fn = FocalLoss(
                    gamma=2.0,  # Focus on hard examples
                    alpha=alpha,  # Balance positive/negative
                    from_logits=False
                )
                logger.info("Using Focal Loss (alpha=%.3f, gamma=2.0)", alpha)

            elif loss_type == 'bfce':  # binary focal cross-entropy
                # Convert pos_weight to alpha for focal loss
                pos_weight = class_weight[1]
                alpha = pos_weight / (1.0 + pos_weight)

                # Use BinaryFocalCrossentropy
                loss_fn = keras.losses.BinaryFocalCrossentropy(
                    apply_class_balancing=True,
                    alpha=alpha,
                    gamma=2.0,
                )
                logger.info("Using BinaryFocalCrossentropy Loss (alpha=%.3f, gamma=2.0)", alpha)

            elif loss_type == 'bce_dice':  # BCE + Dice loss
                loss_fn = BCEDiceLoss()
                logger.info("Using BCE + Dice Loss")

            elif loss_type == 'bce_jaccard':  # BCE + Jaccard loss
                loss_fn = BCEJaccardLoss()
                logger.info("Using BCE + Jaccard Loss")

            elif loss_type == 'tversky':  # Tversky Loss
                loss_fn = TverskyLoss()
                logger.info("Using Tversky Loss")

            elif loss_type == 'f1':  # F1 Loss
                loss_fn = F1Loss()
                logger.info("Using F1 Loss")

            elif loss_type == 'focal_tversky':  # Focal Tversky Loss
                pos_weight = class_weight[1]
                alpha = pos_weight / (1.0 + pos_weight)
                loss_fn = FocalTverskyLoss(alpha=alpha)
                logger.info("Using Focal Tversky Loss (alpha=%.3f)", alpha)

            else:
                raise ValueError(f"Loss function '{loss_type}' not recognized for occurrence models.")

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
    def _plot_training_history(hist, dir_plots, show_plots, tag=None):
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
        tag: str
            A tag to add to the file name (prefix).
        """
        now = datetime.datetime.now()

        if tag is not None:
            prefix = f"{tag}_"
        else:
            prefix = ""

        metrics = ['loss', 'csi', 'ROC_AUC', 'PR_AUC']

        for metric in metrics:
            plt.figure(figsize=(10, 5))
            plt.plot(hist.history[metric], label='train')
            plt.plot(hist.history[f'val_{metric}'], label='valid')
            plt.legend()
            if tag is not None:
                plt.title(f'{metric} ({tag})')
            else:
                plt.title(f'{metric}')
            plt.tight_layout()
            plt.savefig(f'{dir_plots}/{prefix}{metric}_'
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
                logger.info("\nEpoch %s: early stopping due to %s falling below %s for %s consecutive epochs.",
                            epoch + 1, self.monitor, self.min_value, self.patience)
        else:
            self.wait = 0


@tf.keras.utils.register_keras_serializable()
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


@tf.keras.utils.register_keras_serializable()
class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance in binary classification.

    From: Lin et al. (2017) "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002

    Focal loss applies a modulating term to the cross entropy loss in order to
    focus learning on hard misclassified examples. It is particularly effective
    for addressing class imbalance by down-weighting the loss assigned to
    well-classified examples.

    Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the correct class.

    Parameters
    ----------
    gamma : float
        Focusing parameter (default 2.0). Higher values increase focus on hard examples.
        gamma=0 reduces to standard cross-entropy.
    alpha : float or None
        Weight for positive class (0-1). If None, computed from pos_weight.
    pos_weight : float
        Alternative to alpha: multiplicative weight for positive class.
    from_logits : bool
        If True, apply sigmoid to y_pred first.
    """
    def __init__(self, gamma=2.0, alpha=None, pos_weight=None,
                 from_logits=False, name='focal_loss'):
        super().__init__(name=name)
        self.gamma = float(gamma)
        self.from_logits = bool(from_logits)

        # Handle alpha vs pos_weight
        if alpha is not None:
            self.alpha = float(alpha)
        elif pos_weight is not None:
            # Convert pos_weight to alpha (0-1 scale)
            pw = float(pos_weight)
            self.alpha = pw / (1.0 + pw)
        else:
            self.alpha = 0.5  # Balanced

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

        # Clip predictions to avoid log(0)
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Focal loss formulation
        # For positive samples: -alpha * (1-p)^gamma * log(p)
        # For negative samples: -(1-alpha) * p^gamma * log(1-p)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1.0 - y_pred)
        focal_weight = tf.pow(1.0 - pt, self.gamma)

        # Binary cross-entropy
        bce = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)

        # Apply focal weight and class balance
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * focal_weight * bce

        # Return mean loss per sample
        return tf.reduce_mean(focal_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "from_logits": self.from_logits,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            gamma=config.get("gamma", 2.0),
            alpha=config.get("alpha", 0.5),
            from_logits=config.get("from_logits", False),
            name=config.get("name", "focal_loss")
        )


@tf.keras.utils.register_keras_serializable()
class BCEDiceLoss(keras.losses.Loss):
    def __init__(self, alpha=0.5, eps=1e-7, name="bce_dice_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.eps = eps
        self.bce = keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, y_true, y_pred):
        bce = self.bce(y_true, y_pred)

        # Dice part
        probs = tf.nn.sigmoid(y_pred)
        y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        probs_f = tf.reshape(probs, [-1])

        intersection = tf.reduce_sum(probs_f * y_true_f)
        union = tf.reduce_sum(probs_f) + tf.reduce_sum(y_true_f)

        dice = (2.0 * intersection + self.eps) / (union + self.eps)

        return self.alpha * bce + (1.0 - self.alpha) * (1.0 - dice)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "eps": self.eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            alpha=config.get("alpha", 0.5),
            eps=config.get("eps", 1e-7),
            name=config.get("name", "bce_dice_loss")
        )


@tf.keras.utils.register_keras_serializable()
class BCEJaccardLoss(keras.losses.Loss):
    def __init__(self, alpha=0.5, eps=1e-7, name="bce_jaccard_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.eps = eps
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, y_true, y_pred):
        bce = self.bce(y_true, y_pred)

        probs = tf.nn.sigmoid(y_pred)
        y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        probs_f = tf.reshape(probs, [-1])

        intersection = tf.reduce_sum(probs_f * y_true_f)
        union = tf.reduce_sum(probs_f) + tf.reduce_sum(y_true_f) - intersection

        jaccard = (intersection + self.eps) / (union + self.eps)

        return self.alpha * bce + (1.0 - self.alpha) * (1.0 - jaccard)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "eps": self.eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            alpha=config.get("alpha", 0.5),
            eps=config.get("eps", 1e-7),
            name=config.get("name", "bce_jaccard_loss")
        )


@tf.keras.utils.register_keras_serializable()
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

        y_true = tf.reshape(tf.convert_to_tensor(y_true), [-1])
        y_pred = tf.reshape(tf.convert_to_tensor(y_pred), [-1])

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


@tf.keras.utils.register_keras_serializable()
class F1Score(keras.metrics.Metric):
    """
    F1 Score metric accumulating TP/FP/FN.
    """
    def __init__(self, threshold=0.5, name='F1', dtype=tf.float32):
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

        y_true = tf.reshape(tf.convert_to_tensor(y_true), [-1])
        y_pred = tf.reshape(tf.convert_to_tensor(y_pred), [-1])

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


@tf.keras.utils.register_keras_serializable()
class TverskyLoss(keras.losses.Loss):
    """
    Tversky Loss for binary segmentation/classification.

    The Tversky index is a generalization of the Dice coefficient. It is more flexible
    in allowing different weights for false positives and false negatives.

    From: Salehi et al. (2017) "Tversky loss function for image segmentation using
    3D fully convolutional deep networks"

    Loss = 1 - Tversky_Index

    where Tversky_Index = TP / (TP + alpha*FN + beta*FP)

    When alpha = beta = 0.5, it becomes the Dice coefficient.
    When alpha = beta = 1, it becomes the Jaccard index.

    Parameters
    ----------
    alpha : float
        Weight of false negatives (default 0.5).
        Higher values penalize more aggressively for missed positives.
    beta : float
        Weight of false positives (default 0.5).
        Higher values penalize more aggressively for false alarms.
    eps : float
        Small epsilon value to avoid division by zero (default 1e-7).
    """
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-7, name="tversky_loss"):
        super().__init__(name=name)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def call(self, y_true, y_pred):
        # Ensure correct shapes and types
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if y_true.shape.rank == 1:
            y_true = tf.expand_dims(y_true, axis=-1)
        if y_pred.shape.rank == 1:
            y_pred = tf.expand_dims(y_pred, axis=-1)

        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])

        # Calculate components
        true_pos = tf.reduce_sum(y_true_f * y_pred_f)
        false_neg = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
        false_pos = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)

        # Tversky index
        tversky_index = true_pos / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.eps)

        return 1.0 - tversky_index

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "beta": self.beta,
            "eps": self.eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            alpha=config.get("alpha", 0.5),
            beta=config.get("beta", 0.5),
            eps=config.get("eps", 1e-7),
            name=config.get("name", "tversky_loss")
        )


@tf.keras.utils.register_keras_serializable()
class F1Loss(keras.losses.Loss):
    """
    F1 Loss for direct optimization of F1 score in binary classification.

    This loss approximates the F1 score using a smooth/differentiable formulation
    that allows gradient computation during training. It uses the predictions
    directly (soft targets) rather than hard thresholding.

    Loss ≈ 1 - F1_smooth where F1_smooth = 2*TP / (2*TP + FP + FN)

    TP ≈ sum(y_true * y_pred)  # soft TP
    FP ≈ sum((1 - y_true) * y_pred)  # soft FP
    FN ≈ sum(y_true * (1 - y_pred))  # soft FN

    This formulation preserves gradients for training while still optimizing
    toward F1-like behavior. Note: The decision threshold should be applied
    during evaluation, not in the loss function.

    Parameters
    ----------
    eps : float
        Small epsilon value to avoid division by zero (default 1e-7).
    """
    def __init__(self, eps=1e-7, name="f1_loss"):
        super().__init__(name=name)
        self.eps = float(eps)

    def call(self, y_true, y_pred):
        # Ensure correct shapes and types
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if y_true.shape.rank == 1:
            y_true = tf.expand_dims(y_true, axis=-1)
        if y_pred.shape.rank == 1:
            y_pred = tf.expand_dims(y_pred, axis=-1)

        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])

        # Calculate soft TP, FP, FN using continuous predictions
        # This preserves gradients for backpropagation
        true_pos = tf.reduce_sum(y_true_f * y_pred_f)
        false_pos = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
        false_neg = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))

        # Soft F1 score using continuous approximation
        # F1 = 2*TP / (2*TP + FP + FN)
        f1_smooth = (2.0 * true_pos) / (2.0 * true_pos + false_pos + false_neg + self.eps)

        return 1.0 - f1_smooth

    def get_config(self):
        config = super().get_config()
        config.update({
            "eps": self.eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            eps=config.get("eps", 1e-7),
            name=config.get("name", "f1_loss")
        )


@tf.keras.utils.register_keras_serializable()
class FocalTverskyLoss(keras.losses.Loss):
    """
    Focal Tversky Loss - combines Focal Loss with Tversky Loss.

    This loss combines the focusing mechanism of Focal Loss with the flexibility
    of Tversky Loss, making it particularly effective for imbalanced datasets
    where the F1 score is important.

    Loss = (1 - TverskyIndex)^gamma

    From: Abraham & Khan (2019) "A Novel Focal Tversky Loss Function With Improved
    Attention U-Net for Segmentation of Tumor Lesions"

    Parameters
    ----------
    alpha : float
        Weight of false negatives in Tversky (default 0.5).
    beta : float
        Weight of false positives in Tversky (default 0.5).
    gamma : float
        Focusing parameter (default 1.5).
        Higher values focus more on hard examples.
    eps : float
        Small epsilon value to avoid division by zero (default 1e-7).
    """
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.5, eps=1e-7, name="focal_tversky_loss"):
        super().__init__(name=name)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def call(self, y_true, y_pred):
        # Ensure correct shapes and types
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if y_true.shape.rank == 1:
            y_true = tf.expand_dims(y_true, axis=-1)
        if y_pred.shape.rank == 1:
            y_pred = tf.expand_dims(y_pred, axis=-1)

        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])

        # Calculate components
        true_pos = tf.reduce_sum(y_true_f * y_pred_f)
        false_neg = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
        false_pos = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)

        # Tversky index
        tversky_index = true_pos / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.eps)

        # Focal Tversky Loss with power gamma
        focal_tversky_loss = tf.pow(1.0 - tversky_index, self.gamma)

        return focal_tversky_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "eps": self.eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            alpha=config.get("alpha", 0.5),
            beta=config.get("beta", 0.5),
            gamma=config.get("gamma", 1.5),
            eps=config.get("eps", 1e-7),
            name=config.get("name", "focal_tversky_loss")
        )

