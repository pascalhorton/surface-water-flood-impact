"""
Class to handle the DL options for models based on deep learning.
It is not meant to be used directly, but to be inherited by other classes.
"""
import argparse
import datetime

has_optuna = False
try:
    import optuna
    has_optuna = True
except ImportError:
    pass


class ImpactDlOptions:
    """
    The DL basic options.

    Attributes
    ----------
    parser : argparse.ArgumentParser
        The parser object.
    run_name : str
        Name of the run.
    optimize_with_optuna : str
        Optimize with Optuna.
    optuna_trials_nb : int
        Number of Optuna trials.
    optuna_study_name : str
        Optuna study name.
    target_type : str
        The target type. Options are: 'occurrence', 'damage_ratio'
    factor_neg_reduction: int
        The factor to reduce the number of negatives only for training.
    weight_denominator: int
        The weight denominator to reduce the negative class weights.
    random_state: int|None
        The random state to use for the random number generator.
        Default: 42. Set to None to not set the random seed.
    use_precip: bool
        Whether to use precipitation data (CombiPrecip) or not.
    use_simple_features: bool
        Whether to use simple features (event properties and static attributes) or not.
    simple_feature_classes: list
        The list of simple feature classes to use.
    simple_features: list
        The list of simple features to use.
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
    with_batchnorm_dense: bool
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
        self.parser = argparse.ArgumentParser(description="SWAFI DL")
        self._set_parser_shared_arguments()

        # General options
        self.run_name = None
        self.optimize_with_optuna = None
        self.optuna_trials_nb = None
        self.optuna_study_name = None
        self.target_type = None
        self.factor_neg_reduction = None
        self.weight_denominator = None
        self.random_state = None

        # Data options
        self.use_precip = None
        self.use_simple_features = None
        self.simple_feature_classes = None
        self.simple_features = None
        self.log_transform_precip = None
        self.transform_precip = None
        self.transform_static = None

        # Training options
        self.batch_size = None
        self.epochs = None
        self.learning_rate = None

        # Model options for the dense layers
        self.dropout_rate_dense = None
        self.with_batchnorm_dense = None
        self.nb_dense_layers = None
        self.nb_dense_units = None
        self.nb_dense_units_decreasing = None
        self.inner_activation_dense = None

    def _set_parser_shared_arguments(self):
        """
        Set the parser arguments.
        """
        self.parser.add_argument(
            '--run-name', type=str,
            default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            help='The run name')
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
            '--target-type', type=str, default='occurrence',
            help='The target type. Options are: occurrence, damage_ratio')
        self.parser.add_argument(
            '--factor-neg-reduction', type=int, default=10,
            help='The factor to reduce the number of negatives only for training')
        self.parser.add_argument(
            '--weight-denominator', type=int, default=40,
            help='The weight denominator to reduce the negative class weights')
        self.parser.add_argument(
            '--random-state', type=int, default=42,
            help='The random state to use for the random number generator')
        self.parser.add_argument(
            '--do-not-use-precip', action='store_true',
            help='Do not use precipitation data')
        self.parser.add_argument(
            '--do-not-use-simple-features', action='store_true',
            help='Do not use simple features (event properties and static attributes)')
        self.parser.add_argument(
            '--simple-feature-classes', nargs='+',
            default=['event', 'terrain', 'swf_map', 'flowacc', 'twi'],
            help='The list of simple feature classes to use (e.g. event terrain)')
        self.parser.add_argument(
            '--simple-features', nargs='+',
            default=[],
            help='The list of specific simple features to use (e.g. event:i_max_q).'
                 'If not specified, the default class features will be used.'
                 'If specified, the default class features will be overridden for'
                 'this class only (e.g. event).')
        self.parser.add_argument(
            '--no-log-transform-precip', action='store_true',
            help='Do not log-transform the precipitation')
        self.parser.add_argument(
            '--transform-precip', type=str, default='normalize',
            help='The transformation to apply to the 3D data')
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
            '--dropout-rate-dense', type=float, default=0.2,
            help='The dropout rate for the dense layers')
        self.parser.add_argument(
            '--no-batchnorm-dense', action='store_true',
            help='Do not use batch normalization for the dense layers')
        self.parser.add_argument(
            '--nb-dense-layers', type=int, default=5,
            help='The number of dense layers')
        self.parser.add_argument(
            '--nb-dense-units', type=int, default=256,
            help='The number of dense units')
        self.parser.add_argument(
            '--dense-units-decreasing', action='store_true',
            help='The number of dense units should decrease')
        self.parser.add_argument(
            '--inner-activation-dense', type=str, default='leaky_relu',
            help='The inner activation function for the dense layers')

    def _parse_args(self, args):
        """
        Parse the arguments.
        """
        self.run_name = args.run_name
        self.target_type = args.target_type
        self.optimize_with_optuna = args.optimize_with_optuna
        self.optuna_trials_nb = args.optuna_trials_nb
        self.optuna_study_name = args.optuna_study_name
        self.factor_neg_reduction = args.factor_neg_reduction
        self.weight_denominator = args.weight_denominator
        self.random_state = args.random_state
        self.use_precip = not args.do_not_use_precip
        self.use_simple_features = not args.do_not_use_simple_features
        self.simple_feature_classes = args.simple_feature_classes
        self.simple_features = args.simple_features
        self.log_transform_precip = not args.no_log_transform_precip
        self.transform_precip = args.transform_precip
        self.transform_static = args.transform_static
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.dropout_rate_dense = args.dropout_rate_dense
        self.with_batchnorm_dense = not args.no_batchnorm_dense
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

        if self.use_simple_features:
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
                'batch_size', [16, 32, 64, 128, 256, 512])
        if 'learning_rate' in hp_to_optimize:
            self.learning_rate = trial.suggest_float(
                'learning_rate', 1e-4, 1e-2, log=True)
        if 'dropout_rate_dense' in hp_to_optimize:
            self.dropout_rate_dense = trial.suggest_float(
                'dropout_rate_dense', 0.2, 0.5)
        if 'with_batchnorm_dense' in hp_to_optimize:
            self.with_batchnorm_dense = trial.suggest_categorical(
                'with_batchnorm_dense', [True, False])
        if 'nb_dense_layers' in hp_to_optimize:
            self.nb_dense_layers = trial.suggest_int(
                'nb_dense_layers', 1, 10)
        if 'nb_dense_units' in hp_to_optimize:
            self.nb_dense_units = trial.suggest_int(
                'nb_dense_units', 16, 512)
        if 'nb_dense_units_decreasing' in hp_to_optimize:
            self.nb_dense_units_decreasing = trial.suggest_categorical(
                'nb_dense_units_decreasing', [True, False])
        if 'inner_activation_dense' in hp_to_optimize:
            self.inner_activation_dense = trial.suggest_categorical(
                'inner_activation_dense', ['relu', 'tanh', 'sigmoid', 'softmax',
                                           'elu', 'selu', 'leaky_relu', 'linear'])

        return True

    def _print_shared_options(self, show_optuna_params=False):
        print(f"Options (run {self.run_name}):")
        print("- target_type: ", self.target_type)
        print("- random_state: ", self.random_state)
        print("- factor_neg_reduction: ", self.factor_neg_reduction)
        print("- use_precip: ", self.use_precip)
        print("- use_simple_features: ", self.use_simple_features)

        if self.use_simple_features:
            print("- simple_feature_classes: ", self.simple_feature_classes)
            print("- simple_features: ", self.simple_features)

        if self.optimize_with_optuna:
            print("- optimize_with_optuna: ", self.optimize_with_optuna)
            print("- optuna_study_name: ", self.optuna_study_name)
            print("- optuna_trials_nb: ", self.optuna_trials_nb)
            print("- epochs: ", self.epochs)
            if not show_optuna_params:
                return  # Do not print the other options

        print("- weight_denominator: ", self.weight_denominator)

        if self.use_simple_features:
            print("- transform_static: ", self.transform_static)

        if self.use_precip:
            print("- transform_precip: ", self.transform_precip)
            print("- log_transform_precip: ", self.log_transform_precip)

        print("- batch_size: ", self.batch_size)
        print("- epochs: ", self.epochs)
        print("- learning_rate: ", self.learning_rate)
        print("- dropout_rate_dense: ", self.dropout_rate_dense)
        print("- with_batchnorm_dense: ", self.with_batchnorm_dense)
        print("- nb_dense_layers: ", self.nb_dense_layers)
        print("- nb_dense_units: ", self.nb_dense_units)
        print("- nb_dense_units_decreasing: ", self.nb_dense_units_decreasing)
        print("- inner_activation_dense: ", self.inner_activation_dense)
