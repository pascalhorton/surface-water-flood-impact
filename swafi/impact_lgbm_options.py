"""
Class to handle the LightGBM options.
"""
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


class ImpactLGBMOptions(ImpactBasicOptions):
    """
    The LightGBM options.

    Attributes
    ----------
    weight_denominator: int
        The weight denominator to reduce the negative class weights.
    n_estimators: int
        The number of boosting rounds.
    learning_rate: float
        The learning rate (shrinkage).
    num_leaves: int
        Maximum number of leaves per tree.
    max_depth: int
        Maximum tree depth. -1 means no limit.
    min_child_samples: int
        Minimum number of data points in a leaf.
    subsample: float
        Fraction of samples used per tree.
    colsample_bytree: float
        Fraction of features used per tree.
    reg_alpha: float
        L1 regularization term.
    reg_lambda: float
        L2 regularization term.
    early_stopping_rounds: int
        Stop training if validation metric does not improve for this many rounds.
    """
    def __init__(self):
        super().__init__()
        self._set_parser_lgbm_arguments()

        self.weight_denominator = None
        self.n_estimators = None
        self.learning_rate = None
        self.num_leaves = None
        self.max_depth = None
        self.min_child_samples = None
        self.subsample = None
        self.colsample_bytree = None
        self.reg_alpha = None
        self.reg_lambda = None
        self.early_stopping_rounds = None

    def copy(self):
        return copy.deepcopy(self)

    def _set_parser_lgbm_arguments(self):
        self.parser.add_argument(
            '--weight-denominator', type=int, default=10,
            help='The weight denominator to reduce the negative class weights')
        self.parser.add_argument(
            '--n-estimators', type=int, default=1000,
            help='The number of boosting rounds')
        self.parser.add_argument(
            '--learning-rate', type=float, default=0.05,
            help='The learning rate (shrinkage)')
        self.parser.add_argument(
            '--num-leaves', type=int, default=31,
            help='Maximum number of leaves per tree')
        self.parser.add_argument(
            '--max-depth', type=int, default=-1,
            help='Maximum tree depth (-1 means no limit)')
        self.parser.add_argument(
            '--min-child-samples', type=int, default=20,
            help='Minimum number of data points in a leaf')
        self.parser.add_argument(
            '--subsample', type=float, default=0.8,
            help='Fraction of samples used per tree')
        self.parser.add_argument(
            '--colsample-bytree', type=float, default=0.8,
            help='Fraction of features used per tree')
        self.parser.add_argument(
            '--reg-alpha', type=float, default=0.0,
            help='L1 regularization term')
        self.parser.add_argument(
            '--reg-lambda', type=float, default=1.0,
            help='L2 regularization term')
        self.parser.add_argument(
            '--early-stopping-rounds', type=int, default=50,
            help='Stop if validation metric does not improve for this many rounds')

    def parse_args(self):
        args = self.parser.parse_args()
        self._parse_basic_args(args)
        self.weight_denominator = args.weight_denominator
        self.n_estimators = args.n_estimators
        self.learning_rate = args.learning_rate
        self.num_leaves = args.num_leaves
        self.max_depth = args.max_depth
        self.min_child_samples = args.min_child_samples
        self.subsample = args.subsample
        self.colsample_bytree = args.colsample_bytree
        self.reg_alpha = args.reg_alpha
        self.reg_lambda = args.reg_lambda
        self.early_stopping_rounds = args.early_stopping_rounds

    def generate_for_optuna(self, trial, hp_to_optimize='default'):
        """
        Generate the hyperparameters for Optuna.

        Parameters
        ----------
        trial: optuna.trial.Trial
            The trial.
        hp_to_optimize: list|str
            The hyperparameters to optimize. Can be the string 'default'.
            Options are: 'weight_denominator', 'n_estimators', 'learning_rate',
            'num_leaves', 'max_depth', 'min_child_samples', 'subsample',
            'colsample_bytree', 'reg_alpha', 'reg_lambda'.

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
                'weight_denominator', 'n_estimators', 'learning_rate',
                'num_leaves', 'max_depth', 'min_child_samples',
                'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']

        if 'weight_denominator' in hp_to_optimize:
            self.weight_denominator = trial.suggest_int('weight_denominator', 1, 100)
        if 'n_estimators' in hp_to_optimize:
            self.n_estimators = trial.suggest_int('n_estimators', 100, 3000)
        if 'learning_rate' in hp_to_optimize:
            self.learning_rate = trial.suggest_float('learning_rate', 0.005, 0.3, log=True)
        if 'num_leaves' in hp_to_optimize:
            self.num_leaves = trial.suggest_int('num_leaves', 10, 300)
        if 'max_depth' in hp_to_optimize:
            self.max_depth = trial.suggest_int('max_depth', 3, 15)
        if 'min_child_samples' in hp_to_optimize:
            self.min_child_samples = trial.suggest_int('min_child_samples', 5, 200)
        if 'subsample' in hp_to_optimize:
            self.subsample = trial.suggest_float('subsample', 0.5, 1.0)
        if 'colsample_bytree' in hp_to_optimize:
            self.colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        if 'reg_alpha' in hp_to_optimize:
            self.reg_alpha = trial.suggest_float('reg_alpha', 0.0, 10.0)
        if 'reg_lambda' in hp_to_optimize:
            self.reg_lambda = trial.suggest_float('reg_lambda', 0.0, 10.0)

        return True

    def print_options(self, show_optuna_params=False):
        logger.info("-" * 80)
        self._print_basic_options()

        if self.optimize_with_optuna and not show_optuna_params:
            logger.info("-" * 80)
            return

        logger.info("- weight_denominator:  %s", self.weight_denominator)
        logger.info("- n_estimators:  %s", self.n_estimators)
        logger.info("- learning_rate:  %s", self.learning_rate)
        logger.info("- num_leaves:  %s", self.num_leaves)
        logger.info("- max_depth:  %s", self.max_depth)
        logger.info("- min_child_samples:  %s", self.min_child_samples)
        logger.info("- subsample:  %s", self.subsample)
        logger.info("- colsample_bytree:  %s", self.colsample_bytree)
        logger.info("- reg_alpha:  %s", self.reg_alpha)
        logger.info("- reg_lambda:  %s", self.reg_lambda)
        logger.info("- early_stopping_rounds:  %s", self.early_stopping_rounds)
        logger.info("-" * 80)

    def is_ok(self):
        if not super().is_ok():
            return False

        assert self.weight_denominator > 0, "Invalid weight_denominator"
        assert self.n_estimators > 0, "Invalid n_estimators"
        assert self.learning_rate > 0, "Invalid learning_rate"
        assert self.num_leaves > 0, "Invalid num_leaves"
        assert self.min_child_samples > 0, "Invalid min_child_samples"
        assert 0 < self.subsample <= 1.0, "Invalid subsample"
        assert 0 < self.colsample_bytree <= 1.0, "Invalid colsample_bytree"
        assert self.reg_alpha >= 0, "Invalid reg_alpha"
        assert self.reg_lambda >= 0, "Invalid reg_lambda"
        assert self.early_stopping_rounds > 0, "Invalid early_stopping_rounds"

        return True
