"""
Class to handle the DL options for models based on deep learning.
It is not meant to be used directly, but to be inherited by other classes.
"""
import datetime
import argparse

from swafi.impact_basic_options import ImpactBasicOptions

has_optuna = False
try:
    import optuna
    has_optuna = True
except ImportError:
    pass


class ImpactDlOptions(ImpactBasicOptions):
    """
    The DL shared options.

    Attributes
    ----------
    optimize_with_optuna : str
        Optimize with Optuna.
    optuna_trials_nb : int
        Number of Optuna trials.
    optuna_study_name : str
        Optuna study name.
    factor_neg_reduction: int
        The factor to reduce the number of negatives only for training.
    weight_denominator: int
        The weight denominator to reduce the negative class weights.
    use_precip: bool
        Whether to use precipitation data (CombiPrecip) or not.
    log_transform_precip: bool
        Whether to log-transform the precipitation or not.
    transform_precip: str
        The transformation to apply to the precipitation data.
        Options are: 'standardize', 'normalize'.
    transform_static: str
        The transformation to apply to the static data.
        Options are: 'standardize', 'normalize'.
    batch_size: int
        The batch size.
    epochs: int
        The number of epochs.
    learning_rate: float
        The learning rate.
    dropout_rate_dense: float
        The dropout rate for the dense layers.
    use_batchnorm_dense: bool
        Whether to use batch normalization or not for the dense layers.
    nb_dense_layers: int
        The number of dense layers.
    nb_dense_units: int
        The number of dense units.
    nb_dense_units_decreasing: bool
        Whether the number of dense units should decrease or not.
    inner_activation_dense: str
        The inner activation function for the dense layers.
    """
    def __init__(self):
        super().__init__()
        self._set_parser_dl_shared_arguments()

        # General options
        self.optimize_with_optuna = None
        self.optuna_trials_nb = None
        self.optuna_study_name = None
        self.factor_neg_reduction = None
        self.weight_denominator = None

        # Data options
        self.use_precip = None
        self.log_transform_precip = None
        self.transform_precip = None
        self.transform_static = None

        # Training options
        self.batch_size = None
        self.epochs = None
        self.learning_rate = None

        # Model options for the dense layers
        self.dropout_rate_dense = None
        self.use_batchnorm_dense = None
        self.nb_dense_layers = None
        self.nb_dense_units = None
        self.nb_dense_units_decreasing = None
        self.inner_activation_dense = None

    def _set_parser_dl_shared_arguments(self):
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
            '--factor-neg-reduction', type=int, default=10,
            help='The factor to reduce the number of negatives only for training')
        self.parser.add_argument(
            '--weight-denominator', type=int, default=40,
            help='The weight denominator to reduce the negative class weights')
        self.parser.add_argument(
            '--use-precip', action=argparse.BooleanOptionalAction, default=True,
            help='Do not use precipitation data')
        self.parser.add_argument(
            '--log-transform-precip', action=argparse.BooleanOptionalAction,
            default=True, help='Log-transform the precipitation')
        self.parser.add_argument(
            '--transform-precip', type=str, default='normalize',
            help='The transformation to apply to the precipitation data')
        self.parser.add_argument(
            '--transform-static', type=str, default='standardize',
            help='The transformation to apply to the static data')
        self.parser.add_argument(
            '--batch-size', type=int, default=64,
            help='The batch size')
        self.parser.add_argument(
            '--epochs', type=int, default=100,
            help='The number of epochs')
        self.parser.add_argument(
            '--learning-rate', type=float, default=0.001,
            help='The learning rate')
        self.parser.add_argument(
            '--dropout-rate-dense', type=float, default=0.4,
            help='The dropout rate for the dense layers')
        self.parser.add_argument(
            '--use-batchnorm-dense', action=argparse.BooleanOptionalAction,
            default=True, help='Use batch normalization for the dense layers')
        self.parser.add_argument(
            '--nb-dense-layers', type=int, default=5,
            help='The number of dense layers')
        self.parser.add_argument(
            '--nb-dense-units', type=int, default=256,
            help='The number of dense units')
        self.parser.add_argument(
            '--dense-units-decreasing', action=argparse.BooleanOptionalAction,
            default=False, help='The number of dense units should decrease')
        self.parser.add_argument(
            '--inner-activation-dense', type=str, default='relu',
            help='The inner activation function for the dense layers')

    def _parse_dl_args(self, args):
        """
        Parse the arguments.
        """
        self._parse_basic_args(args)

        self.optimize_with_optuna = args.optimize_with_optuna
        self.optuna_trials_nb = args.optuna_trials_nb
        self.optuna_study_name = args.optuna_study_name
        self.factor_neg_reduction = args.factor_neg_reduction
        self.weight_denominator = args.weight_denominator
        self.use_precip = args.use_precip
        self.log_transform_precip = args.log_transform_precip
        self.transform_precip = args.transform_precip
        self.transform_static = args.transform_static
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.dropout_rate_dense = args.dropout_rate_dense
        self.use_batchnorm_dense = args.use_batchnorm_dense
        self.nb_dense_layers = args.nb_dense_layers
        self.nb_dense_units = args.nb_dense_units
        self.nb_dense_units_decreasing = args.dense_units_decreasing
        self.inner_activation_dense = args.inner_activation_dense

    def _generate_for_optuna(self, trial, hp_to_optimize):
        if not has_optuna:
            raise ValueError("Optuna is not installed")

        assert self.optimize_with_optuna, "Optimize with Optuna is not set to True"

        if 'weight_denominator' in hp_to_optimize:
            self.weight_denominator = trial.suggest_int(
                'weight_denominator', 1, 100)

        if self.use_static_attributes:
            if 'transform_static' in hp_to_optimize:
                self.transform_static = trial.suggest_categorical(
                    'transform_static', ['standardize', 'normalize'])

        if self.use_precip:
            if 'transform_precip' in hp_to_optimize:
                self.transform_precip = trial.suggest_categorical(
                    'transform_precip', ['standardize', 'normalize'])
            if 'log_transform_precip' in hp_to_optimize:
                self.log_transform_precip = trial.suggest_categorical(
                    'log_transform_precip', [True, False])

        if 'batch_size' in hp_to_optimize:
            self.batch_size = trial.suggest_categorical(
                'batch_size', [16, 32, 64, 128, 256])
        if 'learning_rate' in hp_to_optimize:
            self.learning_rate = trial.suggest_float(
                'learning_rate', 5e-4, 3e-3, log=True)
        if 'dropout_rate_dense' in hp_to_optimize:
            self.dropout_rate_dense = trial.suggest_float(
                'dropout_rate_dense', 0.2, 0.5)
        if 'use_batchnorm_dense' in hp_to_optimize:
            self.use_batchnorm_dense = trial.suggest_categorical(
                'use_batchnorm_dense', [True, False])
        if 'nb_dense_layers' in hp_to_optimize:
            self.nb_dense_layers = trial.suggest_int(
                'nb_dense_layers', 1, 10)
        if 'nb_dense_units' in hp_to_optimize:
            self.nb_dense_units = trial.suggest_categorical(
                'nb_dense_units', [32, 64, 128, 256, 512, 1024])
        if 'nb_dense_units_decreasing' in hp_to_optimize:
            self.nb_dense_units_decreasing = trial.suggest_categorical(
                'nb_dense_units_decreasing', [True, False])
        if 'inner_activation_dense' in hp_to_optimize:
            self.inner_activation_dense = trial.suggest_categorical(
                'inner_activation_dense',
                ['relu', 'silu', 'elu', 'selu', 'leaky_relu',
                 'linear', 'gelu', 'softplus'])

        return True

    def _print_shared_options(self, show_optuna_params=False):
        self._print_basic_options()
        print("- factor_neg_reduction: ", self.factor_neg_reduction)
        print("- use_precip: ", self.use_precip)

        if self.optimize_with_optuna:
            print("- optimize_with_optuna: ", self.optimize_with_optuna)
            print("- optuna_study_name: ", self.optuna_study_name)
            print("- optuna_trials_nb: ", self.optuna_trials_nb)
            print("- epochs: ", self.epochs)
            if not show_optuna_params:
                return  # Do not print the other options

        print("- weight_denominator: ", self.weight_denominator)

        if self.use_static_attributes:
            print("- transform_static: ", self.transform_static)

        if self.use_precip:
            print("- transform_precip: ", self.transform_precip)
            print("- log_transform_precip: ", self.log_transform_precip)

        print("- batch_size: ", self.batch_size)
        print("- epochs: ", self.epochs)
        print("- learning_rate: ", self.learning_rate)
        print("- dropout_rate_dense: ", self.dropout_rate_dense)
        print("- use_batchnorm_dense: ", self.use_batchnorm_dense)
        print("- nb_dense_layers: ", self.nb_dense_layers)
        print("- nb_dense_units: ", self.nb_dense_units)
        print("- nb_dense_units_decreasing: ", self.nb_dense_units_decreasing)
        print("- inner_activation_dense: ", self.inner_activation_dense)

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

        assert self.factor_neg_reduction is not None, "factor_neg_reduction is not set"
        assert self.weight_denominator is not None, "weight_denominator is not set"
        assert isinstance(self.use_precip, bool), "use_precip is not set"
        assert isinstance(self.log_transform_precip, bool), "log_transform_precip is not set"
        assert self.transform_precip in ['standardize', 'normalize'], "transform_precip is not set"
        assert self.transform_static in ['standardize', 'normalize'], "transform_static is not set"
        assert self.batch_size is not None, "batch_size is not set"
        assert self.epochs is not None, "epochs is not set"
        assert self.learning_rate is not None, "learning_rate is not set"
        assert self.dropout_rate_dense is not None, "dropout_rate_dense is not set"
        assert isinstance(self.use_batchnorm_dense, bool), "use_batchnorm_dense is not set"
        assert self.nb_dense_layers is not None, "nb_dense_layers is not set"
        assert self.nb_dense_units is not None, "nb_dense_units is not set"
        assert isinstance(self.nb_dense_units_decreasing, bool), "nb_dense_units_decreasing is not set"
        assert self.inner_activation_dense is not None, "inner_activation_dense is not set"

        return True