"""
Class to compute the impact function.
"""

from .impact import Impact

from enum import Enum, auto
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class ImpactRandomForest(Impact):
    """
    The generic Random Forest Impact class.
    """

    class OptimApproach(Enum):
        MANUAL = auto()
        GRID_SEARCH_CV = auto()
        RANDOM_SEARCH_CV = auto()
        AUTO = auto()

    class OptimMetric(Enum):
        F1 = auto()
        F1_WEIGHTED = auto()
        CSI = auto()

    def __init__(self):
        super().__init__()

        # Set default options
        self.optim_approach = self.OptimApproach.AUTO
        self.optim_metric = self.OptimMetric.F1

        # Hyperparameters - set grid search parameters
        self.param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }

        # Hyperparameters - set default parameters
        self.n_estimators = 100
        self.max_depth = 10
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.max_features = None






    def optimize(self, x_train, y_train, x_test, y_test):
        """
        Optimize the hyperparameters of the model.

        Parameters
        ----------
        x_train: np.array
            The training features
        y_train: np.array
            The training labels
        x_test: np.array
            The testing features
        y_test: np.array
            The testing labels
        """
        if self.optim_approach == self.OptimApproach.GRID_SEARCH_CV:
            self._optimize_grid_search_cv(x_train, y_train, x_test, y_test)
        elif self.optim_approach == self.OptimApproach.RANDOM_SEARCH_CV:
            self._optimize_random_search_cv(x_train, y_train, x_test, y_test)
        elif self.optim_approach == self.OptimApproach.AUTO:
            self._optimize_auto(x_train, y_train, x_test, y_test)
        else:
            raise ValueError(f"Unknown optimization approach: {self.optim_approach}")


