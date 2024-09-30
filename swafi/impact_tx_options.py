"""
Class to define the options for the Transformer-based impact function.
"""
from .impact_dl_options import ImpactDlOptions

import copy


class ImpactTransformerOptions(ImpactDlOptions):
    """
    The Transformer Deep Learning Impact class options.

    Attributes
    ----------
    precip_daily_days_before: int
        The number of days before the event to use for the daily precipitation.
    precip_hf_time_step: int
        The time step for the high-frequency precipitation [h].
    precip_hf_days_before: int
        The number of days before the event to use for the high-frequency precipitation.
    precip_hf_days_after: int
        The number of days after the event to use for the high-frequency precipitation.
    combined_transformer: bool
        Whether to use a combined transformer or one for each precipitation type.
    nb_transformer_blocks_combined: int
        The number of transformer blocks for the combined transformer.
    num_heads_combined: int
        The number of heads for the combined transformer.
    ff_dim_combined: int
        The feed-forward dimension for the combined transformer.
    dropout_rate_combined: float
        The dropout rate for the combined transformer.
    nb_transformer_blocks_daily: int
        The number of transformer blocks for the daily precipitation.
    num_heads_daily: int
        The number of heads for the daily precipitation.
    ff_dim_daily: int
        The feed-forward dimension for the daily precipitation.
    dropout_rate_daily: float
        The dropout rate for the daily precipitation.
    nb_transformer_blocks_high_freq: int
        The number of transformer blocks for the high-frequency precipitation.
    num_heads_high_freq: int
        The number of heads for the high-frequency precipitation.
    ff_dim_high_freq: int
        The feed-forward dimension for the high-frequency precipitation.
    dropout_rate_high_freq: float
        The dropout rate for the high-frequency precipitation.
    nb_transformer_blocks_attributes: int
        The number of transformer blocks for the attributes.
    num_heads_attributes: int
        The number of heads for the attributes.
    ff_dim_attributes: int
        The feed-forward dimension for the attributes.
    dropout_rate_attributes: float
        The dropout rate for the attributes.
    """
    def __init__(self):
        super().__init__()
        self._set_parser_arguments()

        # Data options
        self.precip_daily_days_before = None
        self.precip_hf_time_step = None
        self.precip_hf_days_before = None
        self.precip_hf_days_after = None

        # Transformer options
        self.combined_transformer = None
        self.nb_transformer_blocks_combined = None
        self.num_heads_combined = None
        self.ff_dim_combined = None
        self.dropout_rate_combined = None
        self.nb_transformer_blocks_daily = None
        self.num_heads_daily = None
        self.ff_dim_daily = None
        self.dropout_rate_daily = None
        self.nb_transformer_blocks_high_freq = None
        self.num_heads_high_freq = None
        self.ff_dim_high_freq = None
        self.dropout_rate_high_freq = None
        self.nb_transformer_blocks_attributes = None
        self.num_heads_attributes = None
        self.ff_dim_attributes = None
        self.dropout_rate_attributes = None

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
            "--precip_daily_days_before", type=int, default=30,
            help="The number of days before the event to use for the daily precipitation.")
        self.parser.add_argument(
            "--precip_hf_time_step", type=int, default=60,
            help="The time step for the high-frequency precipitation [min].")
        self.parser.add_argument(
            "--precip_hf_days_before", type=int, default=1,
            help="The number of days before the event to use for the high-frequency precipitation.")
        self.parser.add_argument(
            "--precip_hf_days_after", type=int, default=1,
            help="The number of days after the event to use for the high-frequency precipitation.")
        self.parser.add_argument(
            "--combined_transformer", action="store_true",
            help="Whether to use a combined transformer or one for each precipitation type.")
        self.parser.add_argument(
            "--nb_transformer_blocks_combined", type=int, default=2,
            help="The number of transformer blocks for the combined transformer.")
        self.parser.add_argument(
            "--num_heads_combined", type=int, default=2,
            help="The number of heads for the combined transformer.")
        self.parser.add_argument(
            "--ff_dim_combined", type=int, default=32,
            help="The feed-forward dimension for the combined transformer.")
        self.parser.add_argument(
            "--dropout_rate_combined", type=float, default=0.2,
            help="The dropout rate for the combined transformer.")
        self.parser.add_argument(
            "--nb_transformer_blocks_daily", type=int, default=2,
            help="The number of transformer blocks for the daily precipitation.")
        self.parser.add_argument(
            "--num_heads_daily", type=int, default=2,
            help="The number of heads for the daily precipitation.")
        self.parser.add_argument(
            "--ff_dim_daily", type=int, default=32,
            help="The feed-forward dimension for the daily precipitation.")
        self.parser.add_argument(
            "--dropout_rate_daily", type=float, default=0.2,
            help="The dropout rate for the daily precipitation.")
        self.parser.add_argument(
            "--nb_transformer_blocks_high_freq", type=int, default=2,
            help="The number of transformer blocks for the high-frequency precipitation.")
        self.parser.add_argument(
            "--num_heads_high_freq", type=int, default=2,
            help="The number of heads for the high-frequency precipitation.")
        self.parser.add_argument(
            "--ff_dim_high_freq", type=int, default=32,
            help="The feed-forward dimension for the high-frequency precipitation.")
        self.parser.add_argument(
            "--dropout_rate_high_freq", type=float, default=0.2,
            help="The dropout rate for the high-frequency precipitation.")
        self.parser.add_argument(
            "--nb_transformer_blocks_attributes", type=int, default=2,
            help="The number of transformer blocks for the attributes.")
        self.parser.add_argument(
            "--num_heads_attributes", type=int, default=2,
            help="The number of heads for the attributes.")
        self.parser.add_argument(
            "--ff_dim_attributes", type=int, default=32,
            help="The feed-forward dimension for the attributes.")
        self.parser.add_argument(
            "--dropout_rate_attributes", type=float, default=0.2,
            help="The dropout rate for the attributes.")

    def parse_args(self):
        """
        Parse the arguments.
        """
        args = self.parser.parse_args()
        self._parse_args(args)

        self.precip_daily_days_before = args.precip_daily_days_before
        self.precip_hf_time_step = args.precip_hf_time_step
        self.precip_hf_days_before = args.precip_hf_days_before
        self.precip_hf_days_after = args.precip_hf_days_after
        self.combined_transformer = args.combined_transformer
        self.nb_transformer_blocks_combined = args.nb_transformer_blocks_combined
        self.num_heads_combined = args.num_heads_combined
        self.ff_dim_combined = args.ff_dim_combined
        self.dropout_rate_combined = args.dropout_rate_combined
        self.nb_transformer_blocks_daily = args.nb_transformer_blocks_daily
        self.num_heads_daily = args.num_heads_daily
        self.ff_dim_daily = args.ff_dim_daily
        self.dropout_rate_daily = args.dropout_rate_daily
        self.nb_transformer_blocks_high_freq = args.nb_transformer_blocks_high_freq
        self.num_heads_high_freq = args.num_heads_high_freq
        self.ff_dim_high_freq = args.ff_dim_high_freq
        self.dropout_rate_high_freq = args.dropout_rate_high_freq
        self.nb_transformer_blocks_attributes = args.nb_transformer_blocks_attributes
        self.num_heads_attributes = args.num_heads_attributes
        self.ff_dim_attributes = args.ff_dim_attributes
        self.dropout_rate_attributes = args.dropout_rate_attributes

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
            Options are: 'precip_daily_days_before', 'precip_hf_time_step',
            'precip_hf_days_before', 'precip_hf_days_after', 'combined_transformer',
            'nb_transformer_blocks_combined', 'num_heads_combined', 'ff_dim_combined',
            'dropout_rate_combined', 'nb_transformer_blocks_daily', 'num_heads_daily',
            'ff_dim_daily', 'dropout_rate_daily', 'nb_transformer_blocks_high_freq',
            'num_heads_high_freq', 'ff_dim_high_freq', 'dropout_rate_high_freq',
            'nb_transformer_blocks_attributes', 'num_heads_attributes',
            'ff_dim_attributes', 'dropout_rate_attributes', 'dropout_rate_dense',
            'inner_activation_dense', 'with_batchnorm_dense', 'batch_size',
            'learning_rate'

        Returns
        -------
        ImpactCnnOptions
            The options.
        """
        if isinstance(hp_to_optimize, str) and hp_to_optimize == 'default':
            if self.use_precip:
                hp_to_optimize = [
                    'transform_precip', 'log_transform_precip',
                    'precip_daily_days_before', 'precip_hf_time_step',
                    'precip_hf_days_before', 'precip_hf_days_after',
                    'combined_transformer', 'nb_transformer_blocks_combined',
                    'num_heads_combined', 'ff_dim_combined', 'dropout_rate_combined',
                    'nb_transformer_blocks_daily', 'num_heads_daily', 'ff_dim_daily',
                    'dropout_rate_daily', 'nb_transformer_blocks_high_freq',
                    'num_heads_high_freq', 'ff_dim_high_freq', 'dropout_rate_high_freq',
                    'nb_transformer_blocks_attributes', 'num_heads_attributes',
                    'ff_dim_attributes', 'dropout_rate_attributes',
                    'dropout_rate_dense', 'inner_activation_dense',
                    'with_batchnorm_dense', 'batch_size', 'learning_rate']
            else:
                hp_to_optimize = [
                    'batch_size', 'dropout_rate_dense', 'nb_dense_layers',
                    'nb_dense_units', 'inner_activation_dense',
                    'with_batchnorm_dense', 'learning_rate']

        self._generate_for_optuna(trial, hp_to_optimize)

        if self.use_precip:
            self.precip_daily_days_before = trial.suggest_int(
                "precip_daily_days_before", 1, 60)
            self.precip_hf_time_step = trial.suggest_categorical(
                "precip_hf_time_step", [60, 120, 240, 360, 720])
            self.precip_hf_days_before = trial.suggest_int(
                "precip_hf_days_before", 0, 5)
            self.precip_hf_days_after = trial.suggest_int(
                "precip_hf_days_after", 0, 1)

            self.combined_transformer = trial.suggest_categorical(
                "combined_transformer", [True, False])
            if self.combined_transformer:
                self.nb_transformer_blocks_combined = trial.suggest_int(
                    "nb_transformer_blocks_combined", 1, 4)
                self.num_heads_combined = trial.suggest_int(
                    "num_heads_combined", 1, 8)
                self.ff_dim_combined = trial.suggest_int(
                    "ff_dim_combined", 16, 128)
                self.dropout_rate_combined = trial.suggest_uniform(
                    "dropout_rate_combined", 0.1, 0.5)
            else:
                self.nb_transformer_blocks_daily = trial.suggest_int(
                    "nb_transformer_blocks_daily", 1, 4)
                self.num_heads_daily = trial.suggest_int(
                    "num_heads_daily", 1, 8)
                self.ff_dim_daily = trial.suggest_int(
                    "ff_dim_daily", 16, 128)
                self.dropout_rate_daily = trial.suggest_uniform(
                    "dropout_rate_daily", 0.1, 0.5)

                self.nb_transformer_blocks_high_freq = trial.suggest_int(
                    "nb_transformer_blocks_high_freq", 1, 4)
                self.num_heads_high_freq = trial.suggest_int(
                    "num_heads_high_freq", 1, 8)
                self.ff_dim_high_freq = trial.suggest_int(
                    "ff_dim_high_freq", 16, 128)
                self.dropout_rate_high_freq = trial.suggest_uniform(
                    "dropout_rate_high_freq", 0.1, 0.5)

        if not self.combined_transformer:
            self.nb_transformer_blocks_attributes = trial.suggest_int(
                "nb_transformer_blocks_attributes", 1, 4)
            self.num_heads_attributes = trial.suggest_int(
                "num_heads_attributes", 1, 8)
            self.ff_dim_attributes = trial.suggest_int(
                "ff_dim_attributes", 16, 128)
            self.dropout_rate_attributes = trial.suggest_uniform(
                "dropout_rate_attributes", 0.1, 0.5)

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
        print("Transformer-specific options:")

        if self.optimize_with_optuna and not show_optuna_params:
            print("-" * 80)
            return

        if self.use_precip:
            print("- precip_daily_days_before:", self.precip_daily_days_before)
            print("- precip_hf_time_step:", self.precip_hf_time_step)
            print("- precip_hf_days_before:", self.precip_hf_days_before)
            print("- precip_hf_days_after:", self.precip_hf_days_after)
            print("- combined_transformer:", self.combined_transformer)

            if self.combined_transformer:
                print("- nb_transformer_blocks_combined:", self.nb_transformer_blocks_combined)
                print("- num_heads_combined:", self.num_heads_combined)
                print("- ff_dim_combined:", self.ff_dim_combined)
                print("- dropout_rate_combined:", self.dropout_rate_combined)
            else:
                print("- nb_transformer_blocks_daily:", self.nb_transformer_blocks_daily)
                print("- num_heads_daily:", self.num_heads_daily)
                print("- ff_dim_daily:", self.ff_dim_daily)
                print("- dropout_rate_daily:", self.dropout_rate_daily)

                print("- nb_transformer_blocks_high_freq:", self.nb_transformer_blocks_high_freq)
                print("- num_heads_high_freq:", self.num_heads_high_freq)
                print("- ff_dim_high_freq:", self.ff_dim_high_freq)
                print("- dropout_rate_high_freq:", self.dropout_rate_high_freq)

        if not self.combined_transformer:
            print("- nb_transformer_blocks_attributes:", self.nb_transformer_blocks_attributes)
            print("- num_heads_attributes:", self.num_heads_attributes)
            print("- ff_dim_attributes:", self.ff_dim_attributes)
            print("- dropout_rate_attributes:", self.dropout_rate_attributes)

        print("-" * 80)

    def is_ok(self):
        """
        Check if the options are ok.

        Returns
        -------
        bool
            Whether the options are ok or not.
        """
        # No specific checks for the Transformer options
        return True
