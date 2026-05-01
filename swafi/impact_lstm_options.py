"""
Class to define the options for the LSTM+attention impact function.
"""
import argparse
import copy
import logging
import keras

from swafi.impact_dl_options import ImpactDlOptions


logger = logging.getLogger(__name__)


@keras.saving.register_keras_serializable(package="swafi")
class ImpactLstmOptions(ImpactDlOptions):
    """
    LSTM + multi-head attention impact model options.

    Precipitation is always loaded at a single pixel (precip_window_size=1,
    precip_resolution=1). The time series is fed directly to the LSTM.

    Attributes
    ----------
    use_dem: bool
        Whether to include the local DEM value as a second channel.
    precip_time_step: int
        Precipitation time step [h].
    precip_days_before: int
        Number of days before the event to include.
    precip_days_after: int
        Number of days after the event to include.
    lstm_units: int
        Number of hidden units per LSTM layer.
    lstm_nb_layers: int
        Number of stacked LSTM layers.
    attention_heads: int
        Number of attention heads in MultiHeadAttention.
    attention_key_dim: int
        Key/query dimensionality per attention head.
    dropout_rate_lstm: float
        Dropout applied after each LSTM layer.
    """
    def __init__(self, options_csv=None):
        super().__init__()
        self._set_parser_arguments()

        self.use_dem = None
        self.precip_time_step = None
        self.precip_days_before = None
        self.precip_days_after = None
        self.lstm_units = None
        self.lstm_nb_layers = None
        self.attention_heads = None
        self.attention_key_dim = None
        self.dropout_rate_lstm = None

        if options_csv is not None:
            self.load_from_csv(options_csv)
            if not self.is_ok():
                raise ValueError("Options are not ok.")

    def copy(self):
        return copy.deepcopy(self)

    def get_config(self):
        skip_keys = {"parser"}
        cfg = {}
        for k, v in self.__dict__.items():
            if k.startswith('_') or k in skip_keys:
                continue
            if isinstance(v, (type(None), bool, int, float, str, list, tuple, dict)):
                cfg[k] = list(v) if isinstance(v, tuple) else v
            else:
                cfg[k] = repr(v)
        return cfg

    @classmethod
    def from_config(cls, config):
        obj = cls()
        for k, v in config.items():
            try:
                setattr(obj, k, v)
            except Exception:
                pass
        return obj

    def _set_parser_arguments(self):
        self.parser.add_argument(
            '--use-dem',
            action=argparse.BooleanOptionalAction,
            default=False,
            help='Include the local DEM value as a second input channel'
        )
        self.parser.add_argument(
            '--precip-time-step',
            type=int,
            default=1,
            help='Precipitation time step [h]'
        )
        self.parser.add_argument(
            '--precip-days-before',
            type=int,
            default=4,
            help='Number of days before the event to include'
        )
        self.parser.add_argument(
            '--precip-days-after',
            type=int,
            default=1,
            help='Number of days after the event to include'
        )
        self.parser.add_argument(
            '--lstm-units',
            type=int,
            default=64,
            help='Number of hidden units per LSTM layer'
        )
        self.parser.add_argument(
            '--lstm-nb-layers',
            type=int,
            default=2,
            help='Number of stacked LSTM layers'
        )
        self.parser.add_argument(
            '--attention-heads',
            type=int,
            default=4,
            help='Number of attention heads'
        )
        self.parser.add_argument(
            '--attention-key-dim',
            type=int,
            default=16,
            help='Key/query dimensionality per attention head'
        )
        self.parser.add_argument(
            '--dropout-rate-lstm',
            type=float,
            default=0.2,
            help='Dropout rate applied after each LSTM layer'
        )

    def parse_args(self):
        args = self.parser.parse_args()
        self._parse_dl_args(args)

        self.use_dem = args.use_dem
        self.precip_time_step = args.precip_time_step
        self.precip_days_before = args.precip_days_before
        self.precip_days_after = args.precip_days_after
        self.lstm_units = args.lstm_units
        self.lstm_nb_layers = args.lstm_nb_layers
        self.attention_heads = args.attention_heads
        self.attention_key_dim = args.attention_key_dim
        self.dropout_rate_lstm = args.dropout_rate_lstm

        if not self.use_precip:
            self._apply_ann_mode_defaults(args)

        if self.optimize_with_optuna:
            logger.info("Optimizing with Optuna; some options will be ignored.")

    def generate_for_optuna(self, trial, hp_to_optimize='default'):
        """
        Generate options for an Optuna trial.

        Parameters
        ----------
        trial: optuna.Trial
            The trial.
        hp_to_optimize: list|str
            Hyperparameters to optimize. 'default' uses a standard set.

        Returns
        -------
        bool
        """
        if isinstance(hp_to_optimize, str) and hp_to_optimize == 'default':
            hp_to_optimize = [
                'precip_days_before',
                'log_transform_precip',
                'lstm_units',
                'lstm_nb_layers',
                'attention_heads',
                'attention_key_dim',
                'dropout_rate_lstm',
                'nb_dense_layers',
                'nb_dense_units',
                'nb_dense_units_decreasing',
                'inner_activation_dense',
                'dropout_rate_dense',
                'use_batchnorm_dense',
                'use_layernorm_dense',
                'use_residual_dense',
                'batch_size',
                'learning_rate',
                'weight_denominator',
            ]

        self._generate_for_optuna(trial, hp_to_optimize)

        if 'precip_days_before' in hp_to_optimize:
            self.precip_days_before = trial.suggest_int('precip_days_before', 1, 10)
        if 'precip_days_after' in hp_to_optimize:
            self.precip_days_after = trial.suggest_int('precip_days_after', 1, 2)
        if 'lstm_units' in hp_to_optimize:
            self.lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128, 256])
        if 'lstm_nb_layers' in hp_to_optimize:
            self.lstm_nb_layers = trial.suggest_int('lstm_nb_layers', 1, 4)
        if 'attention_heads' in hp_to_optimize:
            self.attention_heads = trial.suggest_categorical('attention_heads', [1, 2, 4, 8])
        if 'attention_key_dim' in hp_to_optimize:
            self.attention_key_dim = trial.suggest_categorical('attention_key_dim', [8, 16, 32, 64])
        if 'dropout_rate_lstm' in hp_to_optimize:
            self.dropout_rate_lstm = trial.suggest_float('dropout_rate_lstm', 0.0, 0.4)

        return True

    def print_options(self, show_optuna_params=False):
        logger.info("-" * 80)
        self._print_shared_options(show_optuna_params)
        logger.info("LSTM-specific options:")
        logger.info("- use_dem:  %s", self.use_dem)
        if self.use_precip:
            logger.info("- precip_time_step:  %s", self.precip_time_step)
            logger.info("- precip_days_before:  %s", self.precip_days_before)
            logger.info("- precip_days_after:  %s", self.precip_days_after)
            logger.info("- lstm_units:  %s", self.lstm_units)
            logger.info("- lstm_nb_layers:  %s", self.lstm_nb_layers)
            logger.info("- attention_heads:  %s", self.attention_heads)
            logger.info("- attention_key_dim:  %s", self.attention_key_dim)
            logger.info("- dropout_rate_lstm:  %s", self.dropout_rate_lstm)
        logger.info("-" * 80)

    def is_ok(self):
        if not super().is_ok():
            return False

        if self.use_precip:
            assert self.precip_days_before >= 0, "precip_days_before must be >= 0"
            assert self.precip_days_after >= 0, "precip_days_after must be >= 0"
            assert self.precip_time_step > 0, "precip_time_step must be > 0"
            assert self.lstm_units > 0, "lstm_units must be > 0"
            assert self.lstm_nb_layers > 0, "lstm_nb_layers must be > 0"
            assert self.attention_heads > 0, "attention_heads must be > 0"
            assert self.attention_key_dim > 0, "attention_key_dim must be > 0"
            assert 0.0 <= self.dropout_rate_lstm < 1.0, "dropout_rate_lstm must be in [0, 1)"

        return True
