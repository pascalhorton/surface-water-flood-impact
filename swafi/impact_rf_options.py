"""
Class to handle the RF options.
"""
import datetime
import copy
import logging

from swafi.impact_basic_options import ImpactBasicOptions

has_optuna = False
try:
    import optuna
    has_optuna = True
except ImportError:
    pass


logger = logging.getLogger(__name__)


class ImpactRFOptions(ImpactBasicOptions):
    """
    The RF options.

    Attributes
    ----------
    weight_denominator: int
        The weight denominator to reduce the negative class weights.
    n_estimators: int
        The number of estimators.
    criterion: str
        The function to measure the quality of a split. Supported criteria are
        'gini', 'log_loss', and 'entropy'.
    max_depth: int
        The maximum depth.
    min_samples_split: int
        The minimum number of samples to split.
    min_samples_leaf: int
        The minimum number of samples in a leaf.
    max_features: str
        The maximum number of features
    n_jobs: int
        The number of jobs to run in parallel (-1 uses all processors).
    """

    # Valid split criteria per target type (sklearn RF classifier/regressor)
    CLASSIFIER_CRITERIA = ['gini', 'log_loss', 'entropy']
    REGRESSOR_CRITERIA = ['squared_error', 'absolute_error',
                          'friedman_mse', 'poisson']

    def __init__(self):
        super().__init__()
        self._set_parser_rf_arguments()

        # General options
        self.weight_denominator = None

        # RF options
        self.n_estimators = None
        self.criterion = None
        self.max_depth = None
        self.min_samples_split = None
        self.min_samples_leaf = None
        self.max_features = None
        self.n_jobs = None

    def copy(self):
        """
        Make a copy of the object.
        Returns
        -------
        ImpactRFOptions
            The copy of the object.
        """
        return copy.deepcopy(self)

    def _set_parser_rf_arguments(self):
        """
        Set the parser arguments.
        """
        self.parser.add_argument(
            '--weight-denominator', type=int, default=30,
            help='The weight denominator to reduce the negative class weights')
        self.parser.add_argument(
            '--n-estimators', type=int, default=800,
            help='The number of estimators')
        self.parser.add_argument(
            '--criterion', type=str, default=None,
            help='The function to measure the quality of a split. For occurrence: '
                 '\'gini\', \'log_loss\', \'entropy\' (default \'entropy\'). For '
                 'damage_ratio: \'squared_error\', \'absolute_error\', '
                 '\'friedman_mse\', \'poisson\' (default \'squared_error\')')
        self.parser.add_argument(
            '--max-depth', type=int, default=30,
            help='The maximum depth')
        self.parser.add_argument(
            '--min-samples-split', type=int, default=30,
            help='The minimum number of samples to split')
        self.parser.add_argument(
            '--min-samples-leaf', type=int, default=90,
            help='The minimum number of samples in a leaf')
        self.parser.add_argument(
            '--max-features', type=float, default=0.3,
            help='The maximum number of features')
        self.parser.add_argument(
            '--n-jobs', type=int, default=5,
            help='The number of jobs to run in parallel (-1 uses all processors)')

    def parse_args(self):
        """
        Parse the arguments.
        """
        args = self.parser.parse_args()
        self._parse_basic_args(args)

        self.weight_denominator = args.weight_denominator
        self.n_estimators = args.n_estimators
        self.criterion = args.criterion
        self.max_depth = args.max_depth
        self.min_samples_split = args.min_samples_split
        self.min_samples_leaf = args.min_samples_leaf
        self.max_features = args.max_features
        self.n_jobs = args.n_jobs

        # Resolve the default criterion depending on the target type
        if self.criterion is None:
            self.criterion = ('entropy' if self.target_type == 'occurrence'
                              else 'squared_error')

    def generate_for_optuna(self, trial, hp_to_optimize='default'):
        """
        Generate the hyperparameters for Optuna.

        Parameters
        ----------
        trial: optuna.trial.Trial
            The trial.
        hp_to_optimize: list
            The hyperparameters to optimize. Can be the string 'default'.
            Options are: 'weight_denominator', 'n_estimators', 'criterion',
            'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'.

            The 'default' set leaves out two hyperparameters that do not help
            the RF under the threshold-free (average precision) objective:
            - 'n_estimators': more trees only lower the ensemble variance, never
              hurting generalisation, so tuning it against a noisy validation
              score can only mis-select. Fix it high instead (default 800).
            - 'weight_denominator': it shifts the probability *level*, i.e. the
              operating point, which the rank-based objective cannot see and
              which tune_probability_threshold() owns downstream anyway.
            Both stay available if listed explicitly, but are off by default.

        Returns
        -------
        bool
            Whether the generation was successful or not.
        """
        if not has_optuna:
            raise ValueError("Optuna is not installed")

        assert self.optimize_with_optuna, "Optimize with Optuna is not set to True"

        if isinstance(hp_to_optimize, str) and hp_to_optimize == 'default':
            hp_to_optimize = [
                'criterion', 'max_depth',
                'min_samples_split', 'min_samples_leaf', 'max_features']

        if 'weight_denominator' in hp_to_optimize:
            self.weight_denominator = trial.suggest_int(
                'weight_denominator', 1, 100)
        if 'n_estimators' in hp_to_optimize:
            self.n_estimators = trial.suggest_int(
                'n_estimators', 50, 1000)
        if 'criterion' in hp_to_optimize:
            criteria = (self.CLASSIFIER_CRITERIA
                        if self.target_type == 'occurrence'
                        else self.REGRESSOR_CRITERIA)
            self.criterion = trial.suggest_categorical('criterion', criteria)
        if 'max_depth' in hp_to_optimize:
            # Trees on this data are fully grown well before depth 40, so the
            # useful (regularising) action is at the low end: log scale.
            self.max_depth = trial.suggest_int(
                'max_depth', 3, 40, log=True)
        if 'min_samples_split' in hp_to_optimize:
            self.min_samples_split = trial.suggest_int(
                'min_samples_split', 2, 50)
        if 'min_samples_leaf' in hp_to_optimize:
            # The main regulariser for a rare positive class: large leaves
            # smooth the probabilities. Wide, log-scaled range so the optimum
            # is not truncated (the previous 1-100 ceiling was hit at 90).
            self.min_samples_leaf = trial.suggest_int(
                'min_samples_leaf', 1, 500, log=True)
        if 'max_features' in hp_to_optimize:
            self.max_features = trial.suggest_float(
                'max_features', 0.1, 1.0)

        return True

    def print_options(self, show_optuna_params=False):
        """
        Print the options.

        Parameters
        ----------
        show_optuna_params: bool
            Whether to show the Optuna parameters or not.
        """
        logger.info("-" * 80)
        self._print_basic_options()

        if self.optimize_with_optuna and not show_optuna_params:
            logger.info("-" * 80)
            return  # Do not print the other options

        logger.info("- weight_denominator:  %s", self.weight_denominator)
        logger.info("- n_estimators:  %s", self.n_estimators)
        logger.info("- criterion:  %s", self.criterion)
        logger.info("- max_depth:  %s", self.max_depth)
        logger.info("- min_samples_split:  %s", self.min_samples_split)
        logger.info("- min_samples_leaf:  %s", self.min_samples_leaf)
        logger.info("- max_features:  %s", self.max_features)
        logger.info("- n_jobs:  %s", self.n_jobs)

        logger.info("-" * 80)

    def is_ok(self):
        """
        Check if the options are ok.

        Returns
        -------
        bool
            Whether the options are ok or not.
        """
        if not super().is_ok():
            return False

        valid_criteria = (self.CLASSIFIER_CRITERIA
                          if self.target_type == 'occurrence'
                          else self.REGRESSOR_CRITERIA)

        assert self.weight_denominator > 0, "Invalid weight_denominator"
        assert self.n_estimators > 0, "Invalid n_estimators"
        assert self.criterion in valid_criteria, "Invalid criterion"
        assert self.min_samples_split > 0, "Invalid min_samples_split"
        assert self.min_samples_leaf > 0, "Invalid min_samples_leaf"

        return True