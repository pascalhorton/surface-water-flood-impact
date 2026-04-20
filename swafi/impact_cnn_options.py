"""
Class to define the options for the CNN-based impact function.
"""
from swafi.impact_dl_options import ImpactDlOptions

import copy
import logging
import math
import argparse
import keras


logger = logging.getLogger(__name__)


@keras.saving.register_keras_serializable(package="swafi")
class ImpactCnnOptions(ImpactDlOptions):
    """
    The CNN Deep Learning Impact class options.

    Attributes
    ----------
    use_dem: bool
        Whether to use DEM data or not.
    optimize_precip_spatial_extent: bool
        Whether to allow the precipitation spatial extent to be optimized.
    optimize_precip_time_step: bool
        Whether to allow the precipitation time step to be optimized.
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
        The dropout rate for the spatial CNN.
    use_spatial_dropout: bool
        Whether to use spatial dropout or not.
    use_batchnorm_cnn: bool
        Whether to use batch normalization or not for the spatial CNN.
    kernel_size_spatial: int
        The kernel size for the spatial convolution.
    nb_filters: int
        The number of filters for the spatial CNN.
    pool_size_spatial: int
        The pool size for the spatial (max) pooling.
    nb_conv_blocks: int
        The number of spatial convolutional blocks.
    inner_activation_cnn: str
        The inner activation function for the CNN and TCN.
    tcn_filters: int
        Number of filters per TCN Conv1D layer.
    tcn_kernel_size: int
        Kernel size for dilated Conv1D in TCN.
    tcn_nb_layers: int
        Number of dilated Conv1D layers in TCN (dilation rates: 1, 2, 4, ...).
    dropout_rate_tcn: float
        Dropout rate after each TCN layer.
    """
    def __init__(self, options_csv=None):
        super().__init__()
        self._set_parser_arguments()

        # Data options
        self.use_dem = None
        self.optimize_precip_spatial_extent = None
        self.optimize_precip_time_step = None
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
        self.nb_filters = None
        self.pool_size_spatial = None
        self.nb_conv_blocks = None
        self.inner_activation_cnn = None

        # TCN options (temporal axis)
        self.tcn_filters = None
        self.tcn_kernel_size = None
        self.tcn_nb_layers = None
        self.dropout_rate_tcn = None

        if options_csv is not None:
            self.load_from_csv(options_csv)
            if not self.is_ok():
                raise ValueError("Options are not ok.")

    def copy(self):
        """
        Make a copy of the object.
        Returns
        -------
        ImpactCnnOptions
            The copy of the object.
        """
        return copy.deepcopy(self)

    def get_config(self):
        """
        Keras serialization hook.
        Return a JSON-serializable config dict of all public option attributes.
        We exclude the argparse parser object and any private ("_" prefixed) attributes.
        """
        skip_keys = {"parser"}
        cfg = {}
        for k, v in self.__dict__.items():
            if k.startswith('_') or k in skip_keys:
                continue
            # Only keep simple JSON-serializable types (None, bool, int, float, str, list, dict)
            if isinstance(v, (type(None), bool, int, float, str, list, tuple, dict)):
                # Convert tuple -> list for JSON friendliness
                cfg[k] = list(v) if isinstance(v, tuple) else v
            else:
                # Fallback to string repr for any unexpected type
                cfg[k] = repr(v)
        return cfg

    @classmethod
    def from_config(cls, config):
        """
        Keras deserialization hook
        """
        obj = cls()
        for k, v in config.items():
            # Only set attributes that exist (forward compatibility if attributes removed)
            try:
                setattr(obj, k, v)
            except Exception:
                pass
        return obj

    def _set_parser_arguments(self):
        """
        Set the parser arguments.
        """
        self.parser.add_argument(
            '--use-dem',
            action=argparse.BooleanOptionalAction,
            default=False,
            help='Use DEM data'
        )
        self.parser.add_argument(
            '--optimize-precip-spatial-extent',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Allow the precipitation spatial extent to be optimized'
        )
        self.parser.add_argument(
            '--optimize-precip-time-step',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Allow the precipitation time step to be optimized'
        )
        self.parser.add_argument(
            '--precip-window-size',
            type=int,
            default=1,
            help='The precipitation window size [km]'
        )
        self.parser.add_argument(
            '--precip-resolution',
            type=int,
            default=1,
            help='The precipitation resolution [km]'
        )
        self.parser.add_argument(
            '--precip-time-step',
            type=int,
            default=1,
            help='The precipitation time step [h]'
        )
        self.parser.add_argument(
            '--precip-days-before',
            type=int,
            default=7,
            help='The number of days before the claim/event to use for the precipitation'
        )
        self.parser.add_argument(
            '--precip-days-after',
            type=int,
            default=1,
            help='The number of days after the claim/event to use for the precipitation'
        )
        self.parser.add_argument(
            '--dropout-rate-cnn',
            type=float,
            default=0.2,
            help='The dropout rate for the CNN'
        )
        self.parser.add_argument(
            '--use-spatial-dropout',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Use spatial dropout'
        )
        self.parser.add_argument(
            '--use-batchnorm-cnn',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Use batch normalization for the CNN'
        )
        self.parser.add_argument(
            '--kernel-size-spatial',
            type=int,
            default=3,
            help='The kernel size for the spatial convolution'
        )
        self.parser.add_argument(
            '--nb-filters',
            type=int,
            default=32,
            help='The number of filters'
        )
        self.parser.add_argument(
            '--pool-size-spatial',
            type=int,
            default=2,
            help='The pool size for the spatial (max) pooling'
        )
        self.parser.add_argument(
            '--nb-conv-blocks',
            type=int,
            default=2,
            help='The number of convolutional blocks'
        )
        self.parser.add_argument(
            '--inner-activation-cnn',
            type=str,
            default='elu',
            help='The inner activation function for the CNN'
        )
        self.parser.add_argument(
            '--tcn-filters',
            type=int,
            default=64,
            help='Number of filters per TCN Conv1D layer'
        )
        self.parser.add_argument(
            '--tcn-kernel-size',
            type=int,
            default=3,
            help='Kernel size for dilated Conv1D in TCN'
        )
        self.parser.add_argument(
            '--tcn-nb-layers',
            type=int,
            default=3,
            help='Number of dilated Conv1D layers in TCN (dilation rates: 1,2,4,...)'
        )
        self.parser.add_argument(
            '--dropout-rate-tcn',
            type=float,
            default=0.1,
            help='Dropout rate after each TCN layer'
        )

    def parse_args(self):
        """
        Parse the arguments.
        """
        args = self.parser.parse_args()
        self._parse_dl_args(args)

        self.use_dem = args.use_dem
        self.optimize_precip_spatial_extent = args.optimize_precip_spatial_extent
        self.optimize_precip_time_step = args.optimize_precip_time_step
        self.precip_window_size = args.precip_window_size
        self.precip_resolution = args.precip_resolution
        self.precip_time_step = args.precip_time_step
        self.precip_days_before = args.precip_days_before
        self.precip_days_after = args.precip_days_after
        self.dropout_rate_cnn = args.dropout_rate_cnn
        self.use_spatial_dropout = args.use_spatial_dropout
        self.use_batchnorm_cnn = args.use_batchnorm_cnn
        self.kernel_size_spatial = args.kernel_size_spatial
        self.nb_filters = args.nb_filters
        self.pool_size_spatial = args.pool_size_spatial
        self.nb_conv_blocks = args.nb_conv_blocks
        self.inner_activation_cnn = args.inner_activation_cnn
        self.tcn_filters = args.tcn_filters
        self.tcn_kernel_size = args.tcn_kernel_size
        self.tcn_nb_layers = args.tcn_nb_layers
        self.dropout_rate_tcn = args.dropout_rate_tcn

        if self.precip_window_size == 1:
            self.kernel_size_spatial = 1
            self.pool_size_spatial = 1
            self.use_spatial_dropout = False

        if self.optimize_with_optuna:
            logger.info("Optimizing with Optuna; some options will be ignored.")

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
            nb_filters, pool_size_spatial, nb_conv_blocks, nb_dense_layers,
            nb_dense_units, nb_dense_units_decreasing, inner_activation_dense,
            inner_activation_cnn, tcn_filters, tcn_kernel_size, tcn_nb_layers,
            dropout_rate_tcn,

        Returns
        -------
        bool
            Whether the generation was successful or not.
        """
        if isinstance(hp_to_optimize, str) and hp_to_optimize == 'default':
            if self.use_precip:
                hp_to_optimize = [
                    'precip_days_before',
                    'log_transform_precip',
                    'nb_conv_blocks',
                    'nb_filters',
                    'inner_activation_cnn',
                    'dropout_rate_cnn',
                    'tcn_filters',
                    'tcn_kernel_size',
                    'tcn_nb_layers',
                    'dropout_rate_tcn',
                    'nb_dense_layers',
                    'nb_dense_units',
                    'nb_dense_units_decreasing',
                    'inner_activation_dense',
                    'dropout_rate_dense',
                    'use_batchnorm_cnn',
                    'use_batchnorm_dense',
                    'batch_size',
                    'learning_rate',
                    'weight_denominator'
                ]

                if self.optimize_precip_spatial_extent:
                    hp_to_optimize.extend([
                        'precip_window_size',
                        'kernel_size_spatial',
                        'pool_size_spatial'
                    ])
                else:
                    self.precip_window_size = 1
                    self.kernel_size_spatial = 1
                    self.pool_size_spatial = 1
                    self.use_spatial_dropout = False

                if self.optimize_precip_time_step:
                    hp_to_optimize.append('precip_time_step')
                else:
                    self.precip_time_step = 1

            else:
                hp_to_optimize = [
                    'nb_dense_layers',
                    'nb_dense_units',
                    'nb_dense_units_decreasing',
                    'inner_activation_dense',
                    'dropout_rate_dense',
                    'batch_size',
                    'learning_rate',
                    'weight_denominator'
                ]

        self._generate_for_optuna(trial, hp_to_optimize)

        if not self.use_precip:
            return True

        if 'precip_window_size' in hp_to_optimize:
            self.precip_window_size = trial.suggest_categorical(
                'precip_window_size', [1, 3, 5, 7])
        if 'precip_resolution' in hp_to_optimize:
            choices = [v for v in [1, 3, 5] if v <= self.precip_window_size]
            precip_resolution_index =  trial.suggest_int(
                'precip_resolution_index', 0, len(choices) - 1)
            self.precip_resolution = choices[precip_resolution_index]
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
            max_val = min(self.precip_window_size / self.precip_resolution, 5)
            choices = [v for v in [1, 3, 5] if v <= max_val]
            kernel_size_spatial_index = trial.suggest_int(
                'kernel_size_spatial_index', 0, len(choices) - 1)
            self.kernel_size_spatial = choices[kernel_size_spatial_index]
        if 'nb_filters' in hp_to_optimize:
            self.nb_filters = trial.suggest_categorical(
                'nb_filters', [32, 64, 128, 256, 512])
        if 'pool_size_spatial' in hp_to_optimize:
            max_val = min(self.precip_window_size / self.precip_resolution, 4)
            self.pool_size_spatial = trial.suggest_int(
                'pool_size_spatial', 1, max_val)
        if 'nb_conv_blocks' in hp_to_optimize:
            max_val = 5
            if self.pool_size_spatial > 1:
                spatial_size = int(self.precip_window_size / self.precip_resolution)
                max_val = min(max_val, math.floor(math.log(spatial_size, self.pool_size_spatial)))
            self.nb_conv_blocks = trial.suggest_int(
                'nb_conv_blocks', 0, max_val)
            if self.nb_conv_blocks == 0:
                self.nb_conv_blocks = 1
        if 'inner_activation_cnn' in hp_to_optimize:
            self.inner_activation_cnn = trial.suggest_categorical(
                'inner_activation_cnn',
                ['relu', 'leaky_relu', 'silu', 'hard_silu', 'softplus', 'mish'])
        if 'tcn_filters' in hp_to_optimize:
            self.tcn_filters = trial.suggest_categorical(
                'tcn_filters', [32, 64, 128, 256])
        if 'tcn_kernel_size' in hp_to_optimize:
            self.tcn_kernel_size = trial.suggest_categorical(
                'tcn_kernel_size', [2, 3, 4, 5])
        if 'tcn_nb_layers' in hp_to_optimize:
            self.tcn_nb_layers = trial.suggest_int(
                'tcn_nb_layers', 2, 6)
        if 'dropout_rate_tcn' in hp_to_optimize:
            self.dropout_rate_tcn = trial.suggest_float(
                'dropout_rate_tcn', 0.0, 0.3)

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
        self._print_shared_options(show_optuna_params)
        logger.info("CNN-specific options:")

        logger.info("- use_dem:  %s", self.use_dem)

        if self.optimize_with_optuna:
            logger.info("- optimize_precip_spatial_extent:  %s", self.optimize_precip_spatial_extent)
            logger.info("- optimize_precip_time_step:  %s", self.optimize_precip_time_step)
            if not show_optuna_params:
                logger.info("-" * 80)
                return

        if self.use_precip:
            logger.info("- precip_window_size:  %s", self.precip_window_size)
            logger.info("- precip_resolution:  %s", self.precip_resolution)
            logger.info("- precip_time_step:  %s", self.precip_time_step)
            logger.info("- precip_days_before:  %s", self.precip_days_before)
            logger.info("- precip_days_after:  %s", self.precip_days_after)
            logger.info("- use_spatial_dropout:  %s", self.use_spatial_dropout)
            logger.info("- dropout_rate_cnn:  %s", self.dropout_rate_cnn)
            logger.info("- use_batchnorm_cnn:  %s", self.use_batchnorm_cnn)
            logger.info("- kernel_size_spatial:  %s", self.kernel_size_spatial)
            logger.info("- nb_filters:  %s", self.nb_filters)
            logger.info("- pool_size_spatial:  %s", self.pool_size_spatial)
            logger.info("- nb_conv_blocks:  %s", self.nb_conv_blocks)
            logger.info("- inner_activation_cnn:  %s", self.inner_activation_cnn)
            logger.info("- tcn_filters:  %s", self.tcn_filters)
            logger.info("- tcn_kernel_size:  %s", self.tcn_kernel_size)
            logger.info("- tcn_nb_layers:  %s", self.tcn_nb_layers)
            logger.info("- dropout_rate_tcn:  %s", self.dropout_rate_tcn)

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
                logger.warning("DEM will not be used as precipitation is not.")

        return True
