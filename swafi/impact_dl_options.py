"""
Class to handle the DL options for models based on deep learning.
It is not meant to be used directly, but to be inherited by other classes.
"""
import datetime
import argparse
import logging

from swafi.impact_basic_options import ImpactBasicOptions

has_optuna = False
try:
    import optuna
    has_optuna = True
except ImportError:
    pass


logger = logging.getLogger(__name__)


class ImpactDlOptions(ImpactBasicOptions):
    """
    The DL shared options.

    Attributes
    ----------
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
    lr_method: str
        The learning rate schedule. Options are: 'constant', 'cosine_decay'.
    jit_compile: bool
        Whether to enable XLA JIT compilation in Keras model.compile.
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
        self.lr_method = None
        self.loss_function = None
        self.jit_compile = False

        # Model options for the dense layers
        self.dropout_rate_dense = None
        self.use_batchnorm_dense = None
        self.use_layernorm_dense = None
        self.use_residual_dense = None
        self.use_feature_class_embedding = None
        self.feature_class_embedding_size = None
        self.nb_dense_layers = None
        self.nb_dense_units = None
        self.nb_dense_units_decreasing = None
        self.inner_activation_dense = None

        # Training performance options
        self.steps_per_execution = 1

        # Checkpoint / resume options
        self.checkpoint_dir = None
        self.resume_training = False

    def _set_parser_dl_shared_arguments(self):
        """
        Set the parser arguments.
        """
        self.parser.add_argument(
            '--factor-neg-reduction',
            type=int,
            default=1,
            help='The factor to reduce the number of negatives only for training'
        )
        self.parser.add_argument(
            '--weight-denominator',
            type=int,
            default=20,
            help='The weight denominator to reduce the negative class weights'
        )
        self.parser.add_argument(
            '--use-precip',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Use precipitation data'
        )
        self.parser.add_argument(
            '--log-transform-precip',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Log-transform the precipitation'
        )
        self.parser.add_argument(
            '--transform-precip',
            type=str,
            default='normalize',
            help='The transformation to apply to the precipitation data'
        )
        self.parser.add_argument(
            '--transform-static',
            type=str,
            default='standardize',
            help='The transformation to apply to the static data'
        )
        self.parser.add_argument(
            '--batch-size',
            type=int,
            default=64,
            help='The batch size'
        )
        self.parser.add_argument(
            '--epochs',
            type=int,
            default=200,
            help='The number of epochs'
        )
        self.parser.add_argument(
            '--learning-rate',
            type=float,
            default=0.001,
            help='The learning rate'
        )
        self.parser.add_argument(
            '--lr-method',
            type=str,
            default='constant',
            choices=['constant', 'cosine_decay'],
            help='Learning rate schedule: constant or cosine_decay'
        )
        self.parser.add_argument(
            '--loss-function',
            type=str,
            default='focal',
            choices=['wbce', 'focal', 'bfce', 'bce_dice', 'bce_jaccard', 'tversky', 'f1', 'focal_tversky'],
            help='Loss function: '
                 'wbce (Weighted Binary Cross-Entropy), '
                 'focal (Focal Loss), '
                 'bfce (Binary Focal Cross-Entropy), '
                 'bce_dice (Binary Cross-Entropy + Dice Loss), '
                 'bce_jaccard (Binary Cross-Entropy + Jaccard Loss), '
                 'tversky (Tversky Loss), '
                 'f1 (F1 Loss), '
                 'focal_tversky (Focal Tversky Loss)'
        )
        self.parser.add_argument(
            '--jit-compile',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Enable XLA JIT compilation in Keras model.compile. Disabled by default because some GPU CNN conv kernels fail to autotune under XLA.'
        )
        self.parser.add_argument(
            '--dropout-rate-dense',
            type=float,
            default=0.1,
            help='The dropout rate for the dense layers'
        )
        self.parser.add_argument(
            '--use-batchnorm-dense',
            action=argparse.BooleanOptionalAction,
            default=False,
            help='Use batch normalization for the dense layers'
        )
        self.parser.add_argument(
            '--use-layernorm-dense',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Use layer normalization (per-sample) for the dense layers instead of batch norm'
        )
        self.parser.add_argument(
            '--use-residual-dense',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Add residual (skip) connections around each dense layer'
        )
        self.parser.add_argument(
            '--use-feature-class-embedding',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Project each feature class through a separate dense layer before the shared block'
        )
        self.parser.add_argument(
            '--feature-class-embedding-size',
            type=int,
            default=32,
            help='Output size of each per-feature-class embedding Dense layer'
        )
        self.parser.add_argument(
            '--nb-dense-layers',
            type=int,
            default=4,
            help='The number of dense layers'
        )
        self.parser.add_argument(
            '--nb-dense-units',
            type=int,
            default=1024,
            help='The number of dense units'
        )
        self.parser.add_argument(
            '--nb-dense-units-decreasing',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='The number of dense units should decrease'
        )
        self.parser.add_argument(
            '--inner-activation-dense',
            type=str,
            default='leaky_relu',
            help='The inner activation function for the dense layers'
        )
        self.parser.add_argument(
            '--steps-per-execution',
            type=int,
            default=1,
            help='Number of training steps per compiled TF function call (reduces Python/TF overhead)'
        )
        self.parser.add_argument(
            '--checkpoint-dir',
            type=str,
            default=None,
            help='Directory for saving training checkpoints after each epoch. '
                 'If None, checkpointing is disabled.'
        )
        self.parser.add_argument(
            '--resume-training',
            action=argparse.BooleanOptionalAction,
            default=False,
            help='Resume training from the latest checkpoint in --checkpoint-dir.'
        )

    def _parse_dl_args(self, args):
        """
        Parse the arguments.
        """
        self._parse_basic_args(args)

        self.factor_neg_reduction = args.factor_neg_reduction
        self.weight_denominator = args.weight_denominator
        self.use_precip = args.use_precip
        self.log_transform_precip = args.log_transform_precip
        self.transform_precip = args.transform_precip
        self.transform_static = args.transform_static
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.lr_method = args.lr_method
        self.loss_function = args.loss_function
        self.jit_compile = args.jit_compile
        self.dropout_rate_dense = args.dropout_rate_dense
        self.use_batchnorm_dense = args.use_batchnorm_dense
        self.use_layernorm_dense = args.use_layernorm_dense
        self.use_residual_dense = args.use_residual_dense
        self.use_feature_class_embedding = args.use_feature_class_embedding
        self.feature_class_embedding_size = args.feature_class_embedding_size
        self.nb_dense_layers = args.nb_dense_layers
        self.nb_dense_units = args.nb_dense_units
        self.steps_per_execution = args.steps_per_execution
        self.nb_dense_units_decreasing = args.nb_dense_units_decreasing
        self.inner_activation_dense = args.inner_activation_dense
        self.checkpoint_dir = args.checkpoint_dir
        self.resume_training = args.resume_training

    def _apply_ann_mode_defaults(self, args):
        """Apply ANN-friendly defaults for options still at their parser default.

        Called by subclasses when use_precip=False so that dense-only networks
        get sensible defaults without changing the CNN defaults.
        """
        overrides = {
            'dropout_rate_dense': 0.1,
            'nb_dense_units': 256,
            'nb_dense_units_decreasing': False,
            'weight_denominator': 1,
            'use_batchnorm_dense': False,
            'use_layernorm_dense': True,
            'use_residual_dense': True,
            'use_feature_class_embedding': True,
        }
        for attr, ann_default in overrides.items():
            if getattr(args, attr) == self.parser.get_default(attr):
                setattr(self, attr, ann_default)

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
                'batch_size', [32, 64, 128, 256, 512, 1024])
        if 'learning_rate' in hp_to_optimize:
            self.learning_rate = trial.suggest_float(
                'learning_rate', 5e-4, 3e-3, log=True)
        if 'dropout_rate_dense' in hp_to_optimize:
            self.dropout_rate_dense = trial.suggest_float(
                'dropout_rate_dense', 0.2, 0.5)
        if 'use_batchnorm_dense' in hp_to_optimize:
            self.use_batchnorm_dense = trial.suggest_categorical(
                'use_batchnorm_dense', [True, False])
        if 'use_layernorm_dense' in hp_to_optimize:
            self.use_layernorm_dense = trial.suggest_categorical(
                'use_layernorm_dense', [True, False])
        if 'use_residual_dense' in hp_to_optimize:
            self.use_residual_dense = trial.suggest_categorical(
                'use_residual_dense', [True, False])
        if 'use_feature_class_embedding' in hp_to_optimize:
            self.use_feature_class_embedding = trial.suggest_categorical(
                'use_feature_class_embedding', [True, False])
        if 'feature_class_embedding_size' in hp_to_optimize:
            self.feature_class_embedding_size = trial.suggest_categorical(
                'feature_class_embedding_size', [16, 32, 64, 128])
        if 'nb_dense_layers' in hp_to_optimize:
            self.nb_dense_layers = trial.suggest_int(
                'nb_dense_layers', 1, 8)
        if 'nb_dense_units' in hp_to_optimize:
            self.nb_dense_units = trial.suggest_categorical(
                'nb_dense_units', [32, 64, 128, 256, 512, 1024, 2048])
        if 'nb_dense_units_decreasing' in hp_to_optimize:
            self.nb_dense_units_decreasing = trial.suggest_categorical(
                'nb_dense_units_decreasing', [True, False])
        if 'inner_activation_dense' in hp_to_optimize:
            self.inner_activation_dense = trial.suggest_categorical(
                'inner_activation_dense',
                ['relu', 'leaky_relu', 'silu', 'hard_silu', 'elu', 'selu',
                 'gelu', 'softplus', 'mish'])

        return True

    def _print_shared_options(self, show_optuna_params=False):
        self._print_basic_options()
        logger.info("- factor_neg_reduction:  %s", self.factor_neg_reduction)
        logger.info("- use_precip:  %s", self.use_precip)

        if self.optimize_with_optuna:
            logger.info("- epochs:  %s", self.epochs)
            if not show_optuna_params:
                return  # Do not print the other options

        logger.info("- weight_denominator:  %s", self.weight_denominator)

        if self.use_static_attributes:
            logger.info("- transform_static:  %s", self.transform_static)

        if self.use_precip:
            logger.info("- transform_precip:  %s", self.transform_precip)
            logger.info("- log_transform_precip:  %s", self.log_transform_precip)

        logger.info("- loss_function:  %s", self.loss_function)
        logger.info("- jit_compile:  %s", self.jit_compile)
        logger.info("- batch_size:  %s", self.batch_size)
        logger.info("- epochs:  %s", self.epochs)
        logger.info("- learning_rate:  %s", self.learning_rate)
        logger.info("- lr_method:  %s", self.lr_method)
        logger.info("- dropout_rate_dense:  %s", self.dropout_rate_dense)
        logger.info("- use_batchnorm_dense:  %s", self.use_batchnorm_dense)
        logger.info("- use_layernorm_dense:  %s", self.use_layernorm_dense)
        logger.info("- use_residual_dense:  %s", self.use_residual_dense)
        logger.info("- use_feature_class_embedding:  %s", self.use_feature_class_embedding)
        logger.info("- feature_class_embedding_size:  %s", self.feature_class_embedding_size)
        logger.info("- nb_dense_layers:  %s", self.nb_dense_layers)
        logger.info("- nb_dense_units:  %s", self.nb_dense_units)
        logger.info("- nb_dense_units_decreasing:  %s", self.nb_dense_units_decreasing)
        logger.info("- inner_activation_dense:  %s", self.inner_activation_dense)
        logger.info("- checkpoint_dir:  %s", self.checkpoint_dir)
        logger.info("- resume_training:  %s", self.resume_training)

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
        assert self.lr_method in ['constant', 'cosine_decay'], "lr_method must be 'constant' or 'cosine_decay'"
        assert isinstance(self.jit_compile, bool), "jit_compile is not set"
        assert self.dropout_rate_dense is not None, "dropout_rate_dense is not set"
        assert isinstance(self.use_batchnorm_dense, bool), "use_batchnorm_dense is not set"
        assert isinstance(self.use_layernorm_dense, bool), "use_layernorm_dense is not set"
        assert isinstance(self.use_residual_dense, bool), "use_residual_dense is not set"
        assert isinstance(self.use_feature_class_embedding, bool), \
            "use_feature_class_embedding is not set"
        assert self.feature_class_embedding_size is not None, \
            "feature_class_embedding_size is not set"
        assert self.nb_dense_layers is not None, "nb_dense_layers is not set"
        assert self.nb_dense_units is not None, "nb_dense_units is not set"
        assert isinstance(self.nb_dense_units_decreasing, bool), "nb_dense_units_decreasing is not set"
        assert self.inner_activation_dense is not None, "inner_activation_dense is not set"

        return True