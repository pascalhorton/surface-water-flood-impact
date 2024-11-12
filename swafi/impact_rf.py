"""
Class to compute the impact function.
"""

from .impact import Impact

import hashlib
import pickle
import optuna
from enum import Enum, auto
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .utils.plotting import plot_random_forest_feature_importance
from .utils.verification import compute_confusion_matrix, compute_score_binary


class ImpactRandomForest(Impact):
    """
    The generic Random Forest Impact class.

    Parameters
    ----------
    events: Events
        The events object.
    target_type: str
        The target type. Options are: 'occurrence', 'damage_ratio'
    random_state: int|None
        The random state to use for the random number generator.
        Default: None. Set to None to not set the random seed.
    reload_trained_models: bool
        Whether to reload the previously trained models or not.
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

    def __init__(self, events, target_type='occurrence', random_state=None,
                 reload_trained_models=False):
        super().__init__(events, target_type=target_type, random_state=random_state)
        self.reload_trained_models = reload_trained_models

        # Set default options
        self.optim_approach = self.OptimApproach.MANUAL
        self.optim_metric = self.OptimMetric.F1

        # Hyperparameters - set grid search parameters
        self.param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }

        # Hyperparameters - set parameter ranges for Optuna
        self.param_ranges = {
            'weight_denominator': (15, 60),
            'n_estimators': (50, 200),
            'max_depth': (10, 25),
            'min_samples_split': (4, 10),
            'min_samples_leaf': (1, 4),
            'max_features': [None, 'sqrt', 'log2']
        }

        # Hyperparameters - set default parameters
        self.n_estimators = 150
        self.max_depth = 15
        self.min_samples_split = 5
        self.min_samples_leaf = 4
        self.max_features = None

    def fit(self, tag=None):
        """
        Optimize the hyperparameters of the model.

        Parameters
        ----------
        tag: str
            The tag to add to the file name.
        """
        if self.optim_approach == self.OptimApproach.MANUAL:
            self._fit_manual()
        elif self.optim_approach == self.OptimApproach.GRID_SEARCH_CV:
            self._fit_grid_search_cv()
        elif self.optim_approach == self.OptimApproach.RANDOM_SEARCH_CV:
            self._fit_random_search_cv()
        elif self.optim_approach == self.OptimApproach.AUTO:
            self._fit_auto(tag)
        else:
            raise ValueError(f"Unknown optimization approach: {self.optim_approach}")

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

    def _fit_manual(self):
        """
        Fit the model with the given hyperparameters.
        """
        tmp_filename = self._create_model_tmp_file_name()
        if (self.reload_trained_models and self.random_state is not None and
                tmp_filename.exists()):
            print(f"Loading model from {tmp_filename}")
            self.model = pickle.load(open(tmp_filename, 'rb'))
        else:
            print(f"Training model and saving to {tmp_filename}")
            self._define_model()
            self.model.fit(self.x_train, self.y_train)
            pickle.dump(self.model, open(tmp_filename, 'wb'))

    def _fit_grid_search_cv(self):
        """
        Fit the model with grid search cross validation.
        """
        self._define_model()

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring=self._get_scoring(),
            n_jobs=self.n_jobs,
            cv=5)
        grid_search.fit(self.x_train, self.y_train)

        # Print best parameters and corresponding accuracy score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)

        self.model = grid_search.best_estimator_
        tmp_filename = self._create_model_tmp_file_name()
        print(f"Saving model to {tmp_filename}")
        pickle.dump(self.model, open(tmp_filename, 'wb'))

    def _fit_random_search_cv(self):
        """
        Fit the model with random search cross validation.
        """
        self._define_model()

        rand_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_grid,
            scoring=self._get_scoring(),
            n_jobs=self.n_jobs,
            cv=5)
        rand_search.fit(self.x_train, self.y_train)

        # Print best parameters and corresponding accuracy score
        print("Best Parameters:", rand_search.best_params_)
        print("Best Score:", rand_search.best_score_)

        self.model = rand_search.best_estimator_
        tmp_filename = self._create_model_tmp_file_name()
        print(f"Saving model to {tmp_filename}")
        pickle.dump(self.model, open(tmp_filename, 'wb'))

    def _fit_auto(self, tag):
        # Define objective function for Optuna
        def objective(trial):
            weight_denominator = trial.suggest_int(
                'weight_denominator',
                self.param_ranges['weight_denominator'][0],
                self.param_ranges['weight_denominator'][1])
            n_estimators = trial.suggest_int(
                'n_estimators',
                self.param_ranges['n_estimators'][0],
                self.param_ranges['n_estimators'][1])
            max_depth = trial.suggest_int(
                'max_depth',
                self.param_ranges['max_depth'][0],
                self.param_ranges['max_depth'][1])
            min_samples_split = trial.suggest_int(
                'min_samples_split',
                self.param_ranges['min_samples_split'][0],
                self.param_ranges['min_samples_split'][1])
            min_samples_leaf = trial.suggest_int(
                'min_samples_leaf',
                self.param_ranges['min_samples_leaf'][0],
                self.param_ranges['min_samples_leaf'][1])
            max_features = trial.suggest_categorical(
                'max_features',
                self.param_ranges['max_features'])

            class_weight = {0: self.weights[0],
                            1: self.weights[1] / weight_denominator}

            if self.target_type == 'occurrence':
                rf = RandomForestClassifier(
                    class_weight=class_weight,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=self.random_state
                )
            elif self.target_type == 'damage_ratio':
                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=self.random_state
                )

            rf.fit(self.x_train, self.y_train)
            y_pred = rf.predict(self.x_valid)

            if self.optim_metric == self.OptimMetric.F1:
                return f1_score(self.y_valid, y_pred)
            elif self.optim_metric == self.OptimMetric.F1_WEIGHTED:
                return f1_score(self.y_valid, y_pred, sample_weight=class_weight)
            elif self.optim_metric == self.OptimMetric.CSI:
                tp, tn, fp, fn = compute_confusion_matrix(self.y_valid, y_pred)
                csi = compute_score_binary('CSI', tp, tn, fp, fn)
                return csi
            else:
                raise ValueError(f"Unknown optimizer metric: {self.optim_metric}")

        # Create a study object and optimize the objective function
        print(f"Optimizing hyperparameters for {self.optim_metric}")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=25)

        # Record the value for the last time
        study_file = self.tmp_dir / f'rf_study_{tag}.pickle'
        pickle.dump(study, open(study_file, "wb"))

        # Print optimization results
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)
        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Train and evaluate the best model on test data
        best_params = study.best_params
        if self.target_type == 'occurrence':
            self.model = RandomForestClassifier(
                **best_params, random_state=self.random_state)
        elif self.target_type == 'damage_ratio':
            self.model = RandomForestRegressor(
                **best_params, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")

        tmp_filename = self._create_model_tmp_file_name()
        print(f"Training best model and saving to {tmp_filename}")
        self.model.fit(self.x_train, self.y_train)
        pickle.dump(self.model, open(tmp_filename, 'wb'))

    def _get_scoring(self):
        """
        Get the scoring function.
        """
        if self.optim_metric == self.OptimMetric.F1:
            scoring = 'f1'
        elif self.optim_metric == self.OptimMetric.F1_WEIGHTED:
            scoring = 'f1_weighted'
        elif self.optim_metric == self.OptimMetric.CSI:
            scoring = 'csi'
        else:
            raise ValueError(f"Unknown optimizer metric: {self.optim_metric}")

        return scoring

    def _define_model(self):
        """
        Define the model.
        """
        if self.target_type == 'occurrence':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs)
        elif self.target_type == 'damage_ratio':
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=self.n_jobs)
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")

    def _create_model_tmp_file_name(self):
        """
        Create the temporary file name for the model.
        """
        tag_model = (
                pickle.dumps(self.df.shape) + pickle.dumps(self.df.columns) +
                pickle.dumps(self.df.iloc[0]) + pickle.dumps(self.features) +
                pickle.dumps(self.optim_metric) + pickle.dumps(self.n_estimators) +
                pickle.dumps(self.max_depth) + pickle.dumps(self.min_samples_split) +
                pickle.dumps(self.min_samples_leaf) + pickle.dumps(self.max_features) +
                pickle.dumps(self.class_weight) + pickle.dumps(self.random_state) +
                pickle.dumps(self.target_type))
        model_hashed_name = f'rf_model_{hashlib.md5(tag_model).hexdigest()}.pickle'
        tmp_filename = self.tmp_dir / model_hashed_name

        return tmp_filename
