"""
Class to define the options for the CNN-based impact function.
"""
from .impact_dl_options import ImpactDlOptions

import copy
import math
import argparse


class ImpactCnnOptions(ImpactDlOptions):
    """
    The CNN Deep Learning Impact class options.

    Attributes
    ----------
    use_dem: bool
        Whether to use DEM data or not.
    precip_window_size: int
        The precipitation window size [km].
    precip_resolution: int
        The precipitation resolution [km].
    precip_time_step: int
        The precipitation time step [h].
    precip_days_before: int
        The number of days before the event to use for the precipitation.
    precip_days_after: int
        The number of days after the event to use for the precipitation.
    dropout_rate_cnn: float
        The dropout rate for the CNN.
    use_spatial_dropout: bool
        Whether to use spatial dropout or not.
    use_batchnorm_cnn: bool
        Whether to use batch normalization or not for the CNN.
    kernel_size_spatial: int
        The kernel size for the spatial convolution.
    kernel_size_temporal: int
        The kernel size for the temporal convolution.
    nb_filters: int
        The number of filters.
    pool_size_spatial: int
        The pool size for the spatial (max) pooling.
    pool_size_temporal: int
        The pool size for the temporal (max) pooling.
    nb_conv_blocks: int
        The number of convolutional blocks.
    inner_activation_cnn: str
        The inner activation function for the CNN.
    """
    def __init__(self):
        super().__init__()
        self._set_parser_arguments()

        # Data options
        self.use_dem = None
        self.precip_window_size = None
        self.precip_resolution = None
        self.precip_time_step = None
        self.precip_days_before = None
        self.precip_days_after = None

        # Model options
        self.dropout_rate_cnn = None
        self.use_spatial_dropout = None
        self.use_batchnorm_cnn = None
        self.kernel_size_spatial = None
        self.kernel_size_temporal = None
        self.nb_filters = None
        self.pool_size_spatial = None
        self.pool_size_temporal = None
        self.nb_conv_blocks = None
        self.inner_activation_cnn = None

    def copy(self):
        """
        Make a copy of the object.
        Returns
        -------
        ImpactCnnOptions
            The copy of the object.
        """
        return copy.deepcopy(self)
    
    def _set_parser_arguments(self):
        """
        Set the parser arguments.
        """
        self.parser.add_argument(
            '--use-dem', action=argparse.BooleanOptionalAction, default=False,
            help='Use DEM data')
        self.parser.add_argument(
            '--precip-window-size', type=int, default=1,
            help='The precipitation window size [km]')
        self.parser.add_argument(
            '--precip-resolution', type=int, default=1,
            help='The precipitation resolution [km]')
        self.parser.add_argument(
            '--precip-time-step', type=int, default=6,
            help='The precipitation time step [h]')
        self.parser.add_argument(
            '--precip-days-before', type=int, default=7,
            help='The number of days before the claim/event to use for the precipitation')
        self.parser.add_argument(
            '--precip-days-after', type=int, default=1,
            help='The number of days after the claim/event to use for the precipitation')
        self.parser.add_argument(
            '--dropout-rate-cnn', type=float, default=0.4,
            help='The dropout rate for the CNN')
        self.parser.add_argument(
            '--use-spatial-dropout', action=argparse.BooleanOptionalAction,
            default=True, help='Use spatial dropout')
        self.parser.add_argument(
            '--use-batchnorm-cnn', action=argparse.BooleanOptionalAction,
            default=True, help='Use batch normalization for the CNN')
        self.parser.add_argument(
            '--kernel-size-spatial', type=int, default=3,
            help='The kernel size for the spatial convolution')
        self.parser.add_argument(
            '--kernel-size-temporal', type=int, default=3,
            help='The kernel size for the temporal convolution')
        self.parser.add_argument(
            '--nb-filters', type=int, default=32,
            help='The number of filters')
        self.parser.add_argument(
            '--pool-size-spatial', type=int, default=1,
            help='The pool size for the spatial (max) pooling')
        self.parser.add_argument(
            '--pool-size-temporal', type=int, default=2,
            help='The pool size for the temporal (max) pooling')
        self.parser.add_argument(
            '--nb-conv-blocks', type=int, default=4,
            help='The number of convolutional blocks')
        self.parser.add_argument(
            '--inner-activation-cnn', type=str, default='elu',
            help='The inner activation function for the CNN')

    def parse_args(self):
        """
        Parse the arguments.
        """
        args = self.parser.parse_args()
        self._parse_dl_args(args)

        self.use_dem = args.use_dem
        self.precip_window_size = args.precip_window_size
        self.precip_resolution = args.precip_resolution
        self.precip_time_step = args.precip_time_step
        self.precip_days_before = args.precip_days_before
        self.precip_days_after = args.precip_days_after
        self.dropout_rate_cnn = args.dropout_rate_cnn
        self.use_spatial_dropout = args.use_spatial_dropout
        self.use_batchnorm_cnn = args.use_batchnorm_cnn
        self.kernel_size_spatial = args.kernel_size_spatial
        self.kernel_size_temporal = args.kernel_size_temporal
        self.nb_filters = args.nb_filters
        self.pool_size_spatial = args.pool_size_spatial
        self.pool_size_temporal = args.pool_size_temporal
        self.nb_conv_blocks = args.nb_conv_blocks
        self.inner_activation_cnn = args.inner_activation_cnn

        if self.optimize_with_optuna:
            print("Optimizing with Optuna; some options will be ignored.")

    def generate_for_optuna(self, trial, hp_to_optimize='default'):
        """
        Generate the options for Optuna.

        Parameters
        ----------
        trial: optuna.Trial
            The trial.
        hp_to_optimize: list|str
            The list of hyperparameters to optimize. Can be the string 'default'
            Options are: weight_denominator, precip_window_size, precip_time_step,
            precip_days_before, precip_resolution, precip_days_after, transform_static,
            transform_precip, log_transform_precip, batch_size, learning_rate,
            dropout_rate_dense, dropout_rate_cnn, use_spatial_dropout,
            use_batchnorm_cnn, use_batchnorm_dense, kernel_size_spatial,
            kernel_size_temporal, nb_filters, pool_size_spatial, pool_size_temporal,
            nb_conv_blocks, nb_dense_layers, nb_dense_units, nb_dense_units_decreasing,
            inner_activation_dense, inner_activation_cnn,

        Returns
        -------
        bool
            Whether the generation was successful or not.
        """
        if isinstance(hp_to_optimize, str) and hp_to_optimize == 'default':
            if self.use_precip:
                hp_to_optimize = [
                    'precip_window_size', 'precip_time_step', 'precip_days_before',
                    'log_transform_precip', 'nb_conv_blocks', 'nb_filters',
                    'kernel_size_spatial', 'kernel_size_temporal', 'pool_size_spatial',
                    'pool_size_temporal', 'inner_activation_cnn', 'dropout_rate_cnn',
                    'use_batchnorm_cnn', 'nb_dense_layers', 'nb_dense_units',
                    'nb_dense_units_decreasing', 'inner_activation_dense',
                    'dropout_rate_dense', 'use_batchnorm_dense', 'batch_size',
                    'learning_rate', 'weight_denominator']
            else:
                hp_to_optimize = [
                    'nb_dense_layers', 'nb_dense_units',
                    'nb_dense_units_decreasing', 'inner_activation_dense',
                    'dropout_rate_dense', 'use_batchnorm_dense', 'batch_size',
                    'learning_rate', 'weight_denominator']

        self._generate_for_optuna(trial, hp_to_optimize)

        if not self.use_precip:
            return True

        if 'precip_window_size' in hp_to_optimize:
            self.precip_window_size = trial.suggest_categorical(
                'precip_window_size', [1, 3, 5, 7])
        if 'precip_resolution' in hp_to_optimize:
            self.precip_resolution = trial.suggest_categorical(
                'precip_resolution', [1, 3, 5])
        if 'precip_time_step' in hp_to_optimize:
            self.precip_time_step = trial.suggest_categorical(
                'precip_time_step', [1, 2, 4, 6, 12, 24])
        if 'precip_days_before' in hp_to_optimize:
            self.precip_days_before = trial.suggest_int(
                'precip_days_before', 1, 10)
        if 'precip_days_after' in hp_to_optimize:
            self.precip_days_after = trial.suggest_int(
                'precip_days_after', 1, 2)
        if 'dropout_rate_cnn' in hp_to_optimize:
            self.dropout_rate_cnn = trial.suggest_float(
                'dropout_rate_cnn', 0.2, 0.5)
        if 'use_spatial_dropout' in hp_to_optimize:
            self.use_spatial_dropout = trial.suggest_categorical(
                'use_spatial_dropout', [True, False])
        if 'use_batchnorm_cnn' in hp_to_optimize:
            self.use_batchnorm_cnn = trial.suggest_categorical(
                'use_batchnorm_cnn', [True, False])
        if 'kernel_size_spatial' in hp_to_optimize:
            self.kernel_size_spatial = trial.suggest_categorical(
                'kernel_size_spatial', [1, 3, 5])
        if 'kernel_size_temporal' in hp_to_optimize:
            self.kernel_size_temporal = trial.suggest_categorical(
                'kernel_size_temporal', [1, 3, 5, 7, 9, 11])
        if 'nb_filters' in hp_to_optimize:
            self.nb_filters = trial.suggest_categorical(
                'nb_filters', [32, 64, 128, 256, 512])
        if 'pool_size_spatial' in hp_to_optimize:
            self.pool_size_spatial = trial.suggest_categorical(
                'pool_size_spatial', [1, 2, 3, 4])
        if 'pool_size_temporal' in hp_to_optimize:
            self.pool_size_temporal = trial.suggest_categorical(
                'pool_size_temporal', [1, 2, 3, 4, 5, 6, 9, 12])
        if 'nb_conv_blocks' in hp_to_optimize:
            self.nb_conv_blocks = trial.suggest_int(
                'nb_conv_blocks', 0, 5)
        if 'inner_activation_cnn' in hp_to_optimize:
            self.inner_activation_cnn = trial.suggest_categorical(
                'inner_activation_cnn',
                ['relu', 'tanh', 'sigmoid', 'silu', 'elu', 'selu', 'leaky_relu',
                 'linear', 'gelu', 'hard_sigmoid', 'hard_silu', 'softplus'])

        # Check the input 3D size vs nb_conv_blocks
        pixels_nb = int(self.precip_window_size / self.precip_resolution)
        time_steps = int((self.precip_days_before + self.precip_days_after + 1) *
                         24 / self.precip_time_step)

        nb_conv_blocks_max = self.nb_conv_blocks
        if self.pool_size_spatial > 1:
            nb_conv_blocks_max = min(
                nb_conv_blocks_max, math.floor(
                    math.log(pixels_nb, self.pool_size_spatial)))
        if self.pool_size_temporal > 1:
            nb_conv_blocks_max = min(
                nb_conv_blocks_max, math.floor(
                    math.log(time_steps, self.pool_size_temporal)))
        if self.nb_conv_blocks > nb_conv_blocks_max:
            return False  # Not valid

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
        self._print_shared_options(show_optuna_params)
        print("CNN-specific options:")

        print("- use_dem: ", self.use_dem)

        if self.optimize_with_optuna and not show_optuna_params:
            print("-" * 80)
            return

        if self.use_precip:
            print("- precip_window_size: ", self.precip_window_size)
            print("- precip_resolution: ", self.precip_resolution)
            print("- precip_time_step: ", self.precip_time_step)
            print("- precip_days_before: ", self.precip_days_before)
            print("- precip_days_after: ", self.precip_days_after)
            print("- use_spatial_dropout: ", self.use_spatial_dropout)
            print("- dropout_rate_cnn: ", self.dropout_rate_cnn)
            print("- use_batchnorm_cnn: ", self.use_batchnorm_cnn)
            print("- kernel_size_spatial: ", self.kernel_size_spatial)
            print("- kernel_size_temporal: ", self.kernel_size_temporal)
            print("- nb_filters: ", self.nb_filters)
            print("- pool_size_spatial: ", self.pool_size_spatial)
            print("- pool_size_temporal: ", self.pool_size_temporal)
            print("- nb_conv_blocks: ", self.nb_conv_blocks)
            print("- inner_activation_cnn: ", self.inner_activation_cnn)

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

        # Check the precipitation parameters
        if self.use_precip:
            assert self.precip_window_size % self.precip_resolution == 0, \
                "precip_window_size must be divisible by precip_resolution"
            assert self.precip_window_size >= self.precip_resolution, \
                "precip_window_size must be >= precip_resolution"
            assert self.precip_days_before >= 0, "precip_days_before must be >= 0"
            assert self.precip_days_after >= 0, "precip_days_after must be >= 0"

        if not self.use_precip:
            if self.use_dem:
                self.use_dem = False
                print("Warning: DEM will not be used as precipitation is not.")

        return True
