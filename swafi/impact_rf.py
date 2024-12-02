"""
Class to compute the impact function.
"""

from .impact import Impact

import hashlib
import pickle
import copy
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

has_optuna = False
try:
    import optuna
    has_optuna = True
except ImportError:
    pass

from .utils.plotting import plot_random_forest_feature_importance
from .utils.verification import compute_confusion_matrix, compute_score_binary


class ImpactRandomForest(Impact):
    """
    The generic Random Forest Impact class.

    Parameters
    ----------
    events: Events
        The events object.
    options: ImpactOptions
        The model options.
    """

    def __init__(self, events, options):
        super().__init__(events, options)

        self.n_jobs = 20

    def copy(self):
        """
        Make a copy of the object.
        Returns
        -------
        ImpactRandomForest
            The copy of the object.
        """
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

        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

        print(f"Model saved: {filename}")

    def _define_model(self):
        """
        Define the model.
        """
        if self.target_type == 'occurrence':
            self.model = RandomForestClassifier(
                n_estimators=self.options.n_estimators,
                max_depth=self.options.max_depth,
                min_samples_split=self.options.min_samples_split,
                min_samples_leaf=self.options.min_samples_leaf,
                max_features=self.options.max_features,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs)
        elif self.target_type == 'damage_ratio':
            self.model = RandomForestRegressor(
                n_estimators=self.options.n_estimators,
                max_depth=self.options.max_depth,
                min_samples_split=self.options.min_samples_split,
                min_samples_leaf=self.options.min_samples_leaf,
                max_features=self.options.max_features,
                random_state=self.random_state,
                n_jobs=self.n_jobs)
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")

    def compute_f1_score(self, x_valid, y_valid):
        """
        Compute the F1 score on the given set.

        Parameters
        ----------
        x_valid: np.array
            The validation features.
        y_valid: np.array
            The validation target.

        Returns
        -------
        float
            The F1 score.
        """
        epsilon = 1e-7  # a small constant to avoid division by zero

        y_pred = self.model.predict(x_valid)

        y_pred_class = (y_pred > 0.5).astype(int)
        tp, tn, fp, fn = compute_confusion_matrix(y_valid, y_pred_class)
        f1 = 2 * tp / (2 * tp + fp + fn + epsilon)

        return f1

    def fit(self):
        """
        Optimize the hyperparameters of the model.
        """
        self._define_model()
        self.model.fit(self.x_train, self.y_train)

    def plot_feature_importance(self, tag, dir_output=None):
        """
        Plot the feature importance.

        Parameters
        ----------
        tag: str
            The tag to add to the file name.
        dir_output: str
            The output directory. If None, it will be shown and not saved.
             (default: None)
        """
        importances = self.model.feature_importances_
        fig_filename = f'feature_importance_mdi_{tag}'
        plot_random_forest_feature_importance(
            self.model, self.features, importances, fig_filename,
            dir_output=dir_output, n_features=20)
