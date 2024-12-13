"""
Class to handle the RF options.
"""
import datetime
import copy

from swafi.impact_basic_options import ImpactBasicOptions

has_optuna = False
try:
    import optuna
    has_optuna = True
except ImportError:
    pass


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
    """
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
            '--weight-denominator', type=int, default=10,
            help='The weight denominator to reduce the negative class weights')
        self.parser.add_argument(
            '--n-estimators', type=int, default=100,
            help='The number of estimators')
        self.parser.add_argument(
            '--criterion', type=str, default='gini',
            help='The function to measure the quality of a split. Supported criteria are '
                 '\'gini\', \'log_loss\', and \'entropy\'')
        self.parser.add_argument(
            '--max-depth', type=int, default=15,
            help='The maximum depth')
        self.parser.add_argument(
            '--min-samples-split', type=int, default=5,
            help='The minimum number of samples to split')
        self.parser.add_argument(
            '--min-samples-leaf', type=int, default=4,
            help='The minimum number of samples in a leaf')
        self.parser.add_argument(
            '--max-features', type=str, default=None,
            help='The maximum number of features')

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

    def generate_for_optuna(self, trial, hp_to_optimize='default'):
        """
        Generate the hyperparameters for Optuna.

        Parameters
        ----------
        trial: optuna.trial.Trial
            The trial.
        hp_to_optimize: list
            The hyperparameters to optimize. Can be the string 'default'
            Options are: 'weight_denominator', 'n_estimators', 'criterion',
            'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'.

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
                'weight_denominator', 'n_estimators', 'criterion', 'max_depth',
                'min_samples_split', 'min_samples_leaf', 'max_features']

        if 'weight_denominator' in hp_to_optimize:
            self.weight_denominator = trial.suggest_int(
                'weight_denominator', 1, 100)
        if 'n_estimators' in hp_to_optimize:
            self.n_estimators = trial.suggest_int(
                'n_estimators', 50, 1000)
        if 'criterion' in hp_to_optimize:
            self.criterion = trial.suggest_categorical(
                'criterion', ['gini', 'log_loss', 'entropy'])
        if 'max_depth' in hp_to_optimize:
            self.max_depth = trial.suggest_int(
                'max_depth', 5, 100)
        if 'min_samples_split' in hp_to_optimize:
            self.min_samples_split = trial.suggest_int(
                'min_samples_split', 2, 100)
        if 'min_samples_leaf' in hp_to_optimize:
            self.min_samples_leaf = trial.suggest_int(
                'min_samples_leaf', 1, 100)
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
        print("-" * 80)
        self._print_basic_options()

        if self.optimize_with_optuna and not show_optuna_params:
            print("-" * 80)
            return  # Do not print the other options

        print("- weight_denominator: ", self.weight_denominator)
        print("- n_estimators: ", self.n_estimators)
        print("- criterion: ", self.criterion)
        print("- max_depth: ", self.max_depth)
        print("- min_samples_split: ", self.min_samples_split)
        print("- min_samples_leaf: ", self.min_samples_leaf)
        print("- max_features: ", self.max_features)

        print("-" * 80)

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

        assert self.weight_denominator > 0, "Invalid weight_denominator"
        assert self.n_estimators > 0, "Invalid n_estimators"
        assert self.criterion in ['gini', 'log_loss', 'entropy'], "Invalid criterion"
        assert self.min_samples_split > 0, "Invalid min_samples_split"
        assert self.min_samples_leaf > 0, "Invalid min_samples_leaf"
        assert self.max_features in [None, 'sqrt', 'log2'], "Invalid max_features"

        return True