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
    precip_daily_days_nb: int
        The number of days before the event to use for the daily precipitation.
    precip_hf_time_step: int
        The time step for the high-frequency precipitation [h].
    precip_hf_days_before: int
        The number of days before the event to use for the high-frequency precipitation.
    precip_hf_days_after: int
        The number of days after the event to use for the high-frequency precipitation.
    combined_transformer: bool
        Whether to use a combined transformer or one for each precipitation type.
    use_cnn_in_tx: bool
        Whether to use a CNN in the transformer instead of the dense layers.
    nb_transformer_blocks: int
        The number of transformer blocks for the combined transformer.
    tx_model_dim: int
        The model dimension for the combined transformer.
    num_heads: int
        The number of heads for the combined transformer.
    ff_dim: int
        The feed-forward dimension for the combined transformer.
    dropout_rate: float
        The dropout rate for the combined transformer.
    """
    def __init__(self):
        super().__init__()
        self._set_parser_arguments()

        # Data options
        self.precip_daily_days_nb = None
        self.precip_hf_time_step = None
        self.precip_hf_days_before = None
        self.precip_hf_days_after = None

        # Transformer options
        self.combined_transformer = None
        self.embeddings_2_layers = None
        self.embeddings_activation = None
        self.use_cnn_in_tx = None
        self.nb_transformer_blocks = None
        self.tx_model_dim = None
        self.num_heads = None
        self.ff_dim = None
        self.dropout_rate = None

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
            "--precip-daily-days-nb", type=int, default=30,
            help="The number of days to use for the daily precipitation.")
        self.parser.add_argument(
            "--precip-hf-time-step", type=int, default=60,
            help="The time step for the high-frequency precipitation [min].")
        self.parser.add_argument(
            "--precip-hf-days-before", type=int, default=1,
            help="The number of days before the event to use for the high-frequency precipitation.")
        self.parser.add_argument(
            "--precip-hf-days-after", type=int, default=1,
            help="The number of days after the event to use for the high-frequency precipitation.")
        self.parser.add_argument(
            "--combined-transformer", type=bool, default=True,
            help="Whether to use a combined transformer or one for each precipitation type.")
        self.parser.add_argument(
            "--embeddings-2-layers", type=bool, default=False,
            help="Whether to use two dense layers for the embeddings.")
        self.parser.add_argument(
            "--embeddings-activation", type=str, default="relu",
            help="The activation function for the embeddings.")
        self.parser.add_argument(
            "--use-cnn-in-tx", action="store_true",
            help="Whether to use a CNN in the transformer instead of the dense layers.")
        self.parser.add_argument(
            "--nb-transformer-blocks", type=int, default=2,
            help="The number of transformer blocks.")
        self.parser.add_argument(
            "--tx-model-dim", type=int, default=128,
            help="The model dimension.")
        self.parser.add_argument(
            "--num-heads", type=int, default=8,
            help="The number of heads.")
        self.parser.add_argument(
            "--ff-dim", type=int, default=32,
            help="The feed-forward dimension.")
        self.parser.add_argument(
            "--dropout-rate", type=float, default=0.2,
            help="The dropout rate.")

    def parse_args(self):
        """
        Parse the arguments.
        """
        args = self.parser.parse_args()
        self._parse_args(args)

        self.precip_daily_days_nb = args.precip_daily_days_nb
        self.precip_hf_time_step = args.precip_hf_time_step
        self.precip_hf_days_before = args.precip_hf_days_before
        self.precip_hf_days_after = args.precip_hf_days_after
        self.combined_transformer = args.combined_transformer
        self.embeddings_2_layers = args.embeddings_2_layers
        self.embeddings_activation = args.embeddings_activation
        self.use_cnn_in_tx = args.use_cnn_in_tx
        self.nb_transformer_blocks = args.nb_transformer_blocks
        self.tx_model_dim = args.tx_model_dim
        self.num_heads = args.num_heads
        self.ff_dim = args.ff_dim
        self.dropout_rate = args.dropout_rate

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
            Options are: 'precip_daily_days_nb', 'precip_hf_time_step',
            'precip_hf_days_before', 'precip_hf_days_after', 'combined_transformer',
            'embeddings_2_layers', 'embeddings_activation',
            'use_cnn_in_tx', 'nb_transformer_blocks', 'tx_model_dim', 'num_heads',
            'ff_dim', 'dropout_rate', 'dropout_rate_dense', 'inner_activation_dense',
            'with_batchnorm_dense', 'batch_size', 'learning_rate'

        Returns
        -------
        ImpactCnnOptions
            The options.
        """
        if isinstance(hp_to_optimize, str) and hp_to_optimize == 'default':
            if self.use_precip:
                hp_to_optimize = [
                    'transform_precip', 'log_transform_precip',
                    'precip_daily_days_nb', 'precip_hf_time_step',
                    'precip_hf_days_before', 'precip_hf_days_after',
                    'combined_transformer', 'embeddings_2_layers',
                    'embeddings_activation', 'use_cnn_in_tx',
                    'nb_transformer_blocks', 'num_heads', 'ff_dim', 'dropout_rate',
                    'dropout_rate_dense', 'inner_activation_dense',
                    'with_batchnorm_dense', 'batch_size', 'learning_rate']
            else:
                hp_to_optimize = [
                    'batch_size', 'dropout_rate_dense', 'nb_dense_layers',
                    'nb_dense_units', 'inner_activation_dense',
                    'with_batchnorm_dense', 'learning_rate']

        self._generate_for_optuna(trial, hp_to_optimize)

        if self.use_precip:
            if 'precip_daily_days_nb' in hp_to_optimize:
                self.precip_daily_days_nb = trial.suggest_int(
                    "precip_daily_days_nb", 1, 60)
            if 'precip_hf_time_step' in hp_to_optimize:
                self.precip_hf_time_step = trial.suggest_categorical(
                    "precip_hf_time_step", [60, 120, 240, 360, 720])
            if 'precip_hf_days_before' in hp_to_optimize:
                self.precip_hf_days_before = trial.suggest_int(
                    "precip_hf_days_before", 0, 5)
            if 'precip_hf_days_after' in hp_to_optimize:
                self.precip_hf_days_after = trial.suggest_int(
                    "precip_hf_days_after", 0, 1)

        if 'combined_transformer' in hp_to_optimize:
            self.combined_transformer = trial.suggest_categorical(
                "combined_transformer", [True, False])
        if 'use_cnn_in_tx' in hp_to_optimize:
            self.use_cnn_in_tx = trial.suggest_categorical(
                "use_cnn_in_tx", [True, False])
        if 'embeddings_2_layers' in hp_to_optimize:
            self.embeddings_2_layers = trial.suggest_categorical(
                "embeddings_2_layers", [True, False])
        if 'embeddings_activation' in hp_to_optimize:
            self.embeddings_activation = trial.suggest_categorical(
                "embeddings_activation", ["relu", None])
        if 'nb_transformer_blocks' in hp_to_optimize:
            self.nb_transformer_blocks = trial.suggest_int(
                "nb_transformer_blocks", 1, 4)
        if 'tx_model_dim' in hp_to_optimize:
            self.tx_model_dim = trial.suggest_categorical(
                "tx_model_dim", [64, 128, 256, 512, 1024])
        if 'num_heads' in hp_to_optimize:
            self.num_heads = trial.suggest_categorical(
                "num_heads", [1, 2, 4, 8])
        if 'ff_dim' in hp_to_optimize:
            self.ff_dim = trial.suggest_categorical(
                "ff_dim", [16, 32, 64, 128])
        if 'dropout_rate' in hp_to_optimize:
            self.dropout_rate = trial.suggest_uniform(
                "dropout_rate", 0.1, 0.5)

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
            print("- precip_daily_days_nb:", self.precip_daily_days_nb)
            print("- precip_hf_time_step:", self.precip_hf_time_step)
            print("- precip_hf_days_before:", self.precip_hf_days_before)
            print("- precip_hf_days_after:", self.precip_hf_days_after)
            print("- combined_transformer:", self.combined_transformer)
            print("- use_cnn_in_tx:", self.use_cnn_in_tx)

        print("- embeddings_2_layers:", self.embeddings_2_layers)
        print("- embeddings_activation:", self.embeddings_activation)
        print("- nb_transformer_blocks:", self.nb_transformer_blocks)
        print("- tx_model_dim:", self.tx_model_dim)
        print("- num_heads:", self.num_heads)
        print("- ff_dim:", self.ff_dim)
        print("- dropout_rate:", self.dropout_rate)

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
