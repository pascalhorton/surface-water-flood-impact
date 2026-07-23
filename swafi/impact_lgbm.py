"""
Class to compute the impact function with LightGBM.
"""
from .impact import Impact

import copy
import logging
import os
import pickle

has_lightgbm = False
try:
    import lightgbm as lgb
    has_lightgbm = True
except ImportError:
    pass

has_optuna = False
try:
    import optuna
    has_optuna = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


class ImpactLGBM(Impact):
    """
    LightGBM impact model.

    Parameters
    ----------
    options: ImpactLGBMOptions
        The model options.
    events: Events
        The events object.
    """

    def __init__(self, options, events=None):
        super().__init__(options, events)

        if not has_lightgbm:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

    def copy(self):
        return copy.deepcopy(self)

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

        filename = f'{dir_output}/{base_name}_{self.options.run_name}.pkl'

        payload = {
            'model': self.model,
            'features': self.features,
            'probability_threshold': self.probability_threshold,
        }

        # Write to a temporary file then atomically replace, so a reader (or a
        # concurrent writer) never sees a partially written model.
        tmp_filename = f'{filename}.{os.getpid()}.tmp'
        with open(tmp_filename, 'wb') as f:
            pickle.dump(payload, f)
        os.replace(tmp_filename, filename)

        logger.info("Model saved: %s", filename)

    def load_model(self, dir_output, base_name):
        """
        Load the model.

        Parameters
        ----------
        dir_output: str
            The directory where the model is saved.
        base_name: str
            The base name used for the file.
        """
        filename = f'{dir_output}/{base_name}_{self.options.run_name}.pkl'

        with open(filename, 'rb') as f:
            payload = pickle.load(f)

        self.model = payload['model']
        self.features = payload['features']
        self.probability_threshold = payload.get('probability_threshold', 0.5)

        logger.info("Model loaded: %s", filename)

    def fit(self):
        """
        Train the LightGBM model.
        """
        if self.target_type == 'occurrence':
            self.model = lgb.LGBMClassifier(
                objective='binary',
                n_estimators=self.options.n_estimators,
                learning_rate=self.options.learning_rate,
                num_leaves=self.options.num_leaves,
                max_depth=self.options.max_depth,
                min_child_samples=self.options.min_child_samples,
                subsample=self.options.subsample,
                colsample_bytree=self.options.colsample_bytree,
                reg_alpha=self.options.reg_alpha,
                reg_lambda=self.options.reg_lambda,
                class_weight=self.class_weight,
                random_state=self.random_state,
                verbosity=-1,
            )
            callbacks = [
                lgb.early_stopping(self.options.early_stopping_rounds, verbose=True),
                lgb.log_evaluation(100),
            ]
            self.model.fit(
                self.x_train, self.y_train,
                eval_set=[(self.x_valid, self.y_valid)],
                eval_metric='auc',
                callbacks=callbacks,
            )
        elif self.target_type == 'damage_ratio':
            self.model = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=self.options.n_estimators,
                learning_rate=self.options.learning_rate,
                num_leaves=self.options.num_leaves,
                max_depth=self.options.max_depth,
                min_child_samples=self.options.min_child_samples,
                subsample=self.options.subsample,
                colsample_bytree=self.options.colsample_bytree,
                reg_alpha=self.options.reg_alpha,
                reg_lambda=self.options.reg_lambda,
                random_state=self.random_state,
                verbosity=-1,
            )
            callbacks = [
                lgb.early_stopping(self.options.early_stopping_rounds, verbose=True),
                lgb.log_evaluation(100),
            ]
            self.model.fit(
                self.x_train, self.y_train,
                eval_set=[(self.x_valid, self.y_valid)],
                callbacks=callbacks,
            )
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")

    def set_model(self, model):
        """
        Set the model.

        Parameters
        ----------
        model: lgb.LGBMClassifier|lgb.LGBMRegressor
            The model to set.
        """
        self.model = model

    def plot_feature_importance(self, tag, dir_output=None):
        """
        Plot the feature importance.

        Parameters
        ----------
        tag: str
            The tag to add to the file name.
        dir_output: str
            The output directory. If None, it will be shown and not saved.
        """
        if self.model is None:
            raise ValueError("Model not defined")

        try:
            import matplotlib.pyplot as plt
            ax = lgb.plot_importance(self.model, max_num_features=20, importance_type='gain')
            ax.set_title(f'Feature importance ({tag})')
            if dir_output is not None:
                fig_path = f'{dir_output}/feature_importance_lgbm_{tag}.png'
                ax.figure.savefig(fig_path, bbox_inches='tight')
                logger.info("Feature importance saved: %s", fig_path)
            else:
                plt.show()
            plt.close()
        except Exception as e:
            logger.warning("Could not plot feature importance: %s", e)
