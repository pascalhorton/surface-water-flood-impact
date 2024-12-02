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
    optimize_with_optuna : str
        Optimize with Optuna.
    optuna_trials_nb : int
        Number of Optuna trials.
    optuna_study_name : str
        Optuna study name.
    weight_denominator: int
        The weight denominator to reduce the negative class weights.
    n_estimators: int
        The number of estimators.
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
        self.optimize_with_optuna = None
        self.optuna_trials_nb = None
        self.optuna_study_name = None
        self.weight_denominator = None

        # RF options
        self.n_estimators = None
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
            '--optimize-with-optuna', action='store_true',
            help='Optimize the hyperparameters with Optuna')
        self.parser.add_argument(
            '--optuna-trials-nb', type=int, default=100,
            help='The number of trials for Optuna')
        self.parser.add_argument(
            '--optuna-study-name', type=str,
            default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            help='The Optuna study name (default: using the date and time'),
        self.parser.add_argument(
            '--weight-denominator', type=int, default=40,
            help='The weight denominator to reduce the negative class weights')
        self.parser.add_argument(
            '--n-estimators', type=int, default=100,
            help='The number of estimators')
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
            '--max-features', type=str, default='auto',
            help='The maximum number of features')

    def parse_args(self):
        """
        Parse the arguments.
        """
        args = self.parser.parse_args()
        self._parse_basic_args(args)

        self.optimize_with_optuna = args.optimize_with_optuna
        self.optuna_trials_nb = args.optuna_trials_nb
        self.optuna_study_name = args.optuna_study_name
        self.weight_denominator = args.weight_denominator
        self.n_estimators = args.n_estimators
        self.max_depth = args.max_depth
        self.min_samples_split = args.min_samples_split
        self.min_samples_leaf = args.min_samples_leaf
        self.max_features = args.max_features

    def generate_for_optuna(self, trial, hp_to_optimize):
        """
        Generate the hyperparameters for Optuna.

        Parameters
        ----------
        trial: optuna.trial.Trial
            The trial.
        hp_to_optimize: list
            The hyperparameters to optimize.
            Options are: 'weight_denominator', 'n_estimators', 'max_depth',
            'min_samples_split', 'min_samples_leaf', 'max_features'.

        Returns
        -------
        bool
            Whether the generation was successful or not.
        """
        if not has_optuna:
            raise ValueError("Optuna is not installed")

        assert self.optimize_with_optuna, "Optimize with Optuna is not set to True"

        if 'weight_denominator' in hp_to_optimize:
            self.weight_denominator = trial.suggest_int(
                'weight_denominator', 1, 100)
        if 'n_estimators' in hp_to_optimize:
            self.n_estimators = trial.suggest_int(
                'n_estimators', 50, 1000)
        if 'max_depth' in hp_to_optimize:
            self.max_depth = trial.suggest_int(
                'max_depth', 1, 100)
        if 'min_samples_split' in hp_to_optimize:
            self.min_samples_split = trial.suggest_int(
                'min_samples_split', 2, 100)
        if 'min_samples_leaf' in hp_to_optimize:
            self.min_samples_leaf = trial.suggest_int(
                'min_samples_leaf', 1, 100)
        if 'max_features' in hp_to_optimize:
            self.max_features = trial.suggest_categorical(
                'max_features', [None, 'sqrt', 'log2'])

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

        if self.optimize_with_optuna:
            print("- optimize_with_optuna: ", self.optimize_with_optuna)
            print("- optuna_study_name: ", self.optuna_study_name)
            print("- optuna_trials_nb: ", self.optuna_trials_nb)
            if not show_optuna_params:
                print("-" * 80)
                return  # Do not print the other options

        print("- weight_denominator: ", self.weight_denominator)
        print("- n_estimators: ", self.n_estimators)
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
        assert self.min_samples_split > 0, "Invalid min_samples_split"
        assert self.min_samples_leaf > 0, "Invalid min_samples_leaf"
        assert self.max_features in ['auto', 'sqrt', 'log2'], "Invalid max_features"

        return True