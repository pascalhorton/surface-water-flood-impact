"""
Class to define the options for the Transformer-based impact function.
"""
from .impact_dl_options import ImpactDlOptions

import copy
import logging
import argparse


logger = logging.getLogger(__name__)


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
    architecture: str
        Architecture of the Transformer.
    embeddings_2_layers: bool
        Whether to use two dense layers for the embeddings.
    embeddings_activation: str
        The activation function for the embeddings.
    use_single_attributes_vector: bool
        Whether to use a single attributes vector for the Transformer.
    use_precip_type_embedding: bool
        Whether to use a precipitation type embedding.
    inner_activation_tx: str
        The activation function for the Transformer.
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
        self.architecture = None
        self.embeddings_2_layers = None
        self.embeddings_activation = None
        self.use_single_attributes_vector = None
        self.use_precip_type_embedding = None
        self.inner_activation_tx = None
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
            "--precip-daily-days-nb", type=int, default=60,
            help="The number of days to use for the daily precipitation.")
        self.parser.add_argument(
            "--precip-hf-time-step", type=int, default=60,
            help="The time step for the high-frequency precipitation [min].")
        self.parser.add_argument(
            "--precip-hf-days-before", type=int, default=2,
            help="The number of days before the event to use for the high-frequency precipitation.")
        self.parser.add_argument(
            "--precip-hf-days-after", type=int, default=1,
            help="The number of days after the event to use for the high-frequency precipitation.")
        self.parser.add_argument(
            "--architecture", type=str, default="combined_fixed_embeddings",
            help="The architecture of the Transformer.")
        self.parser.add_argument(
            "--embeddings-2-layers", action=argparse.BooleanOptionalAction,
            default=False, help="Whether to use two dense layers for the embeddings.")
        self.parser.add_argument(
            "--embeddings-activation", type=str, default="relu",
            help="The activation function for the embeddings.")
        self.parser.add_argument(
            "--use-single-attributes-vector", action=argparse.BooleanOptionalAction,
            default=False, help="Whether to use a single attributes vector for the Transformer.")
        self.parser.add_argument(
            "--use-precip-type-embedding", action=argparse.BooleanOptionalAction,
            default=True, help="Whether to use a precipitation type embedding.")
        self.parser.add_argument(
            "--inner-activation-tx", type=str, default="relu",
            help="The activation function for the Transformer.")
        self.parser.add_argument(
            "--use-cnn-in-tx", action=argparse.BooleanOptionalAction, default=False,
            help="Whether to use a CNN in the transformer instead of the dense layers.")
        self.parser.add_argument(
            "--nb-transformer-blocks", type=int, default=2,
            help="The number of transformer blocks.")
        self.parser.add_argument(
            "--tx-model-dim", type=int, default=64,
            help="The model dimension.")
        self.parser.add_argument(
            "--num-heads", type=int, default=4,
            help="The number of heads.")
        self.parser.add_argument(
            "--ff-dim", type=int, default=64,
            help="The feed-forward dimension.")
        self.parser.add_argument(
            "--dropout-rate", type=float, default=0.3,
            help="The dropout rate.")

    def parse_args(self):
        """
        Parse the arguments.
        """
        args = self.parser.parse_args()
        self._parse_dl_args(args)

        self.precip_daily_days_nb = args.precip_daily_days_nb
        self.precip_hf_time_step = args.precip_hf_time_step
        self.precip_hf_days_before = args.precip_hf_days_before
        self.precip_hf_days_after = args.precip_hf_days_after
        self.architecture = args.architecture
        self.embeddings_2_layers = args.embeddings_2_layers
        self.embeddings_activation = args.embeddings_activation
        self.use_single_attributes_vector = args.use_single_attributes_vector
        self.use_precip_type_embedding = args.use_precip_type_embedding
        self.inner_activation_tx = args.inner_activation_tx
        self.use_cnn_in_tx = args.use_cnn_in_tx
        self.nb_transformer_blocks = args.nb_transformer_blocks
        self.tx_model_dim = args.tx_model_dim
        self.num_heads = args.num_heads
        self.ff_dim = args.ff_dim
        self.dropout_rate = args.dropout_rate

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
            Options are: 'precip_daily_days_nb', 'precip_hf_time_step',
            'precip_hf_days_before', 'precip_hf_days_after', 'architecture',
            'embeddings_2_layers', 'embeddings_activation', 'inner_activation_tx',
            'use_cnn_in_tx', 'nb_transformer_blocks', 'tx_model_dim', 'num_heads',
            'ff_dim', 'dropout_rate', 'dropout_rate_dense', 'inner_activation_dense',
            'use_batchnorm_dense', 'batch_size', 'learning_rate', 'use_single_attributes_vector',
            'weight_denominator', 'use_precip_type_embedding'.

        Returns
        -------
        bool
            Whether the options are ok or not.
        """
        assert self.use_precip, "Precipitation data is required for the Transformer model."

        if isinstance(hp_to_optimize, str) and hp_to_optimize == 'default':
            hp_to_optimize = [
                'log_transform_precip', 'architecture', 'nb_transformer_blocks',
                'tx_model_dim', 'num_heads', 'inner_activation_tx', 'embeddings_activation',
                'embeddings_2_layers', 'use_single_attributes_vector', 'use_precip_type_embedding',
                'dropout_rate', 'ff_dim', 'nb_dense_layers', 'nb_dense_units',
                'dropout_rate_dense', 'batch_size', 'learning_rate', 'weight_denominator'
                ]

        self._generate_for_optuna(trial, hp_to_optimize)

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
        if 'architecture' in hp_to_optimize:
            self.architecture = trial.suggest_categorical(
                "architecture",
                ['combined_fixed_embeddings', 'combined_learned_embeddings',
                 'combined_broadcast', 'separate', 'hybrid'])
        if 'inner_activation_tx' in hp_to_optimize:
            self.inner_activation_tx = trial.suggest_categorical(
                "inner_activation_tx",
                ['relu', 'leaky_relu', 'silu', 'hard_silu', 'elu', 'selu',
                 'gelu', 'softplus', 'mish'])
        if 'use_single_attributes_vector' in hp_to_optimize:
            # Only available for 'combined_learned_embeddings'. For the other architectures, -1 is used (~None).
            min_val = 0 if self.architecture == 'combined_learned_embeddings' else -1
            max_val = 1 if self.architecture == 'combined_learned_embeddings' else -1
            flag = trial.suggest_int(
                "use_single_attributes_vector", min_val, max_val)
            self.use_single_attributes_vector = flag == 1
        if 'use_precip_type_embedding' in hp_to_optimize:
            # Only available for 'combined_learned_embeddings'. For the other architectures, -1 is used (~None).
            min_val = 0 if self.architecture == 'combined_learned_embeddings' else -1
            max_val = 1 if self.architecture == 'combined_learned_embeddings' else -1
            flag = trial.suggest_int(
                "use_precip_type_embedding", min_val, max_val)
            self.use_precip_type_embedding = flag == 1
        if 'use_cnn_in_tx' in hp_to_optimize:
            self.use_cnn_in_tx = trial.suggest_categorical(
                "use_cnn_in_tx", [True, False])
        if 'embeddings_2_layers' in hp_to_optimize:
            self.embeddings_2_layers = trial.suggest_categorical(
                "embeddings_2_layers", [True, False])
        if 'embeddings_activation' in hp_to_optimize:
            self.embeddings_activation = trial.suggest_categorical(
                "embeddings_activation",
                ['relu', 'leaky_relu', 'silu', 'hard_silu', 'elu', 'selu',
                 'gelu', 'softplus', 'mish'])
        if 'nb_transformer_blocks' in hp_to_optimize:
            self.nb_transformer_blocks = trial.suggest_int(
                "nb_transformer_blocks", 1, 4)
        if 'tx_model_dim' in hp_to_optimize:
            self.tx_model_dim = trial.suggest_categorical(
                "tx_model_dim", [64, 128, 256, 512, 1024])
        if 'num_heads' in hp_to_optimize:
            self.num_heads = trial.suggest_categorical(
                "num_heads", [4, 8, 16])
        if 'ff_dim' in hp_to_optimize:
            self.ff_dim = trial.suggest_categorical(
                "ff_dim", [16, 32, 64, 128])
        if 'dropout_rate' in hp_to_optimize:
            self.dropout_rate = trial.suggest_uniform(
                "dropout_rate", 0.0, 0.4)

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
        logger.info("Transformer-specific options:")

        if self.optimize_with_optuna and not show_optuna_params:
            logger.info("-" * 80)
            return

        logger.info("- precip_daily_days_nb: %s", self.precip_daily_days_nb)
        logger.info("- precip_hf_time_step: %s", self.precip_hf_time_step)
        logger.info("- precip_hf_days_before: %s", self.precip_hf_days_before)
        logger.info("- precip_hf_days_after: %s", self.precip_hf_days_after)
        logger.info("- architecture: %s", self.architecture)
        logger.info("- inner_activation_tx: %s", self.inner_activation_tx)
        logger.info("- use_cnn_in_tx: %s", self.use_cnn_in_tx)
        logger.info("- embeddings_2_layers: %s", self.embeddings_2_layers)
        logger.info("- embeddings_activation: %s", self.embeddings_activation)
        logger.info("- use_single_attributes_vector: %s", self.use_single_attributes_vector)
        logger.info("- use_precip_type_embedding: %s", self.use_precip_type_embedding)
        logger.info("- nb_transformer_blocks: %s", self.nb_transformer_blocks)
        logger.info("- tx_model_dim: %s", self.tx_model_dim)
        logger.info("- num_heads: %s", self.num_heads)
        logger.info("- ff_dim: %s", self.ff_dim)
        logger.info("- dropout_rate: %s", self.dropout_rate)

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

        assert self.use_precip, "Precipitation data is required for the Transformer model."

        return True
