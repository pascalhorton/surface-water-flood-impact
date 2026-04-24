"""
Class to define the options for the Transformer-based impact function.
"""
import argparse
import datetime
import copy
import ast
import logging
import pandas as pd
from typing import List


logger = logging.getLogger(__name__)


class ImpactBasicOptions:
    """
    The basic Impact classes options.

    Attributes
    ----------
    parser : argparse.ArgumentParser
        The parser object.
    run_name: str
        The name of the run.
    dataset: str
        The name of the dataset (mobiliar or gvz).
    event_file_label: str
        The event file label (default: 'default_occurrence').
    event_method: str|None
        The event extraction method. Options: 'simple', 'classic'. Default: None.
    target_type : str
        The target type. Options are: 'occurrence', 'damage_ratio'
    random_state: int|None
        The random state to use for the random number generator.
        Default: None. Set to None to not set the random seed.
    use_event_attributes: bool
        Whether to use event attributes or not.
    use_static_attributes: bool
        Whether to use and static attributes or not.
    use_all_static_attributes: bool
        Whether to use all static attributes or not.
        If not, only the default class features will be used.
    simple_feature_classes: list
        The list of simple feature classes to use.
    replace_simple_features: list
        The list of simple features to use instead of the default ones.
    optimize_with_optuna : str
        Optimize with Optuna.
    optuna_trials_nb : int
        Number of Optuna trials.
    optuna_study_name : str
        Optuna study name.
    optuna_random_sampler : bool
        Use the random sampler for Optuna.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="SWAFI")
        self._set_parser_basic_arguments()

        # Basic options
        self.run_name = None
        self.dataset = None
        self.event_file_label = None
        self.event_method = None
        self.min_nb_claims = None
        self.target_type = None
        self.random_state = None
        self.use_event_attributes = None
        self.use_static_attributes = None
        self.use_all_static_attributes = None
        self.simple_feature_classes = None
        self.replace_simple_features = None

        # Optuna options
        self.optimize_with_optuna = None
        self.optuna_trials_nb = None
        self.optuna_study_name = None
        self.optuna_random_sampler = None

    def copy(self):
        """
        Make a copy of the object.
        Returns
        -------
        ImpactBasicOptions
            The copy of the object.
        """
        return copy.deepcopy(self)

    def load_from_csv(self, options_csv):
        """
        Load the options from a CSV file.

        Parameters
        ----------
        options_csv : str
            The path to the CSV file.
        """
        df = pd.read_csv(options_csv)

        # Parse the arguments to set the default values
        self.parse_args()

        # Set the attributes from the CSV file
        for row in df.itertuples():
            key = row[1]
            val = row[2]

            # Skip some keys
            if key in ['parser', 'run_name', 'dataset']:
                continue

            # Check that the key is valid
            if not hasattr(self, key):
                raise ValueError(f"Unknown option: {key}")

            # Convert the value to the correct type
            attr_type = type(getattr(self, key))
            if key == 'random_state':
                if val in ['None', 'none', 'null', '']:
                    val = None
                else:
                    val = int(val)
            elif attr_type == bool:
                val = val in ['True', 'true', '1', 'yes']
            elif attr_type == int:
                val = int(val)
            elif attr_type == float:
                val = float(val)
            elif attr_type == str:
                val = str(val)
            elif attr_type == list:
                val = self._parse_list_string(val)
            elif attr_type == type(None):
                if val in ['None', 'none', 'null', '']:
                    val = None
                else:
                    raise ValueError(f"Invalid value for NoneType option: {val}")
            elif attr_type == str:
                val = str(val)
            else:
                raise ValueError(f"Unsupported option type: {attr_type}")

            # Set the attribute
            setattr(self, key, val)
    
    def _set_parser_basic_arguments(self):
        """
        Set the parser arguments.
        """
        self.parser.add_argument(
            '--run-name', type=str,
            default=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            help='The run name')
        self.parser.add_argument(
            "--dataset", type=str, default='',
            help="The name of the dataset (mobiliar or gvz).")
        self.parser.add_argument(
            "--event-file-label", type=str, default='default_occurrence',
            help="The event file label (default: 'default_occurrence').")
        self.parser.add_argument(
            "--event-method", type=str, default=None,
            choices=['simple', 'classic'],
            help="The event extraction method ('simple' or 'classic').")
        self.parser.add_argument(
            '--min-nb-claims', type=int, default=1,
            help='The minimum number of claims for an event to be considered.')
        self.parser.add_argument(
            '--target-type', type=str, default='occurrence',
            help='The target type. Options are: occurrence, damage_ratio')
        self.parser.add_argument(
            '--random-state', type=int, default=None,
            help='The random state to use for the random number generator')
        self.parser.add_argument(
            '--use-event-attributes', action=argparse.BooleanOptionalAction,
            default=True, help='Use event attributes (i_max_q, p_sum_q, duration, ...)')
        self.parser.add_argument(
            '--use-static-attributes', action=argparse.BooleanOptionalAction,
            default=True, help='Use static attributes (terrain, swf_map, flowacc, twi)')
        self.parser.add_argument(
            '--use-all-static-attributes', action=argparse.BooleanOptionalAction,
            default=False, help='Use all static attributes.')
        self.parser.add_argument(
            '--simple-feature-classes', nargs='+',
            default=['default'],
            help='The list of simple feature classes to use (e.g. event terrain)')
        self.parser.add_argument(
            '--replace-simple-features', nargs='+',
            default=[],
            help='The list of specific simple features to use (e.g. event:i_max_q).'
                 'If not specified, the default class features will be used.'
                 'If specified, the default class features will be overridden for'
                 'this class only (e.g. event).')
        self.parser.add_argument(
            '--optimize-with-optuna', action='store_true',
            help='Optimize the hyperparameters with Optuna')
        self.parser.add_argument(
            '--optuna-trials-nb', type=int, default=100,
            help='The number of trials for Optuna')
        self.parser.add_argument(
            '--optuna-study-name', type=str,
            default=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            help='The Optuna study name (default: using the date and time'),
        self.parser.add_argument(
            '--optuna-random-sampler', action=argparse.BooleanOptionalAction,
            default=False, help='Use the random sampler for Optuna')

    def parse_args(self):
        """
        Parse the arguments.
        """
        args = self.parser.parse_args()
        self._parse_basic_args(args)

    def _parse_basic_args(self, args):
        """
        Parse the arguments.
        """
        self.run_name = args.run_name
        self.dataset = args.dataset
        self.event_file_label = args.event_file_label
        self.event_method = args.event_method
        self.min_nb_claims = args.min_nb_claims
        self.target_type = args.target_type
        self.random_state = args.random_state
        self.use_event_attributes = args.use_event_attributes
        self.use_static_attributes = args.use_static_attributes
        self.use_all_static_attributes = args.use_all_static_attributes
        if args.simple_feature_classes == ['default']:
            classes = []
            if self.use_event_attributes:
                classes.append('event')
            if self.use_static_attributes:
                classes.extend(['terrain', 'swf_map', 'flowacc', 'twi'])
            self.simple_feature_classes = classes
        elif args.simple_feature_classes == ['none']:
            self.simple_feature_classes = []
        else:
            self.simple_feature_classes = args.simple_feature_classes
        self.replace_simple_features = args.replace_simple_features
        self.optimize_with_optuna = args.optimize_with_optuna
        self.optuna_trials_nb = args.optuna_trials_nb
        self.optuna_study_name = args.optuna_study_name
        self.optuna_random_sampler = args.optuna_random_sampler

    def print_options(self):
        """
        Print the options.
        """
        logger.info("-" * 80)
        self._print_basic_options()
        logger.info("-" * 80)

    def _print_basic_options(self):
        """
        Print the options.
        """
        logger.info("Options (run %s):", self.run_name)
        logger.info("- dataset:  %s", self.dataset)
        logger.info("- event_file_label:  %s", self.event_file_label)
        logger.info("- event_method:  %s", self.event_method)
        logger.info("- min_nb_claims:  %s", self.min_nb_claims)
        logger.info("- target_type:  %s", self.target_type)
        logger.info("- random_state:  %s", self.random_state)
        logger.info("- use_event_attributes:  %s", self.use_event_attributes)
        logger.info("- use_static_attributes:  %s", self.use_static_attributes)
        logger.info("- use_all_static_attributes:  %s", self.use_all_static_attributes)

        if self.use_static_attributes or self.use_event_attributes:
            logger.info("- simple_feature_classes:  %s", self.simple_feature_classes)
            logger.info("- replace simple_features:  %s", self.replace_simple_features)

        if self.optimize_with_optuna:
            logger.info("- optimize_with_optuna:  %s", self.optimize_with_optuna)
            logger.info("- optuna_study_name:  %s", self.optuna_study_name)
            logger.info("- optuna_trials_nb:  %s", self.optuna_trials_nb)
            logger.info("- optuna_random_sampler:  %s", self.optuna_random_sampler)

    def get_attributes_tag(self):
        """
        Get the attributes tag.

        Returns
        -------
        str
            The attributes tag.
        """
        if not self.use_event_attributes and not self.use_static_attributes:
            return 'no_atts'

        if self.use_event_attributes and not self.use_static_attributes:
            return 'event_atts'

        tag = ''
        if self.use_event_attributes:
            tag = 'event_and_'

        if self.use_static_attributes:
            if self.use_all_static_attributes:
                tag += 'all_static_atts'
            else:
                tag += 'static_atts'

        return tag

    def is_ok(self):
        """
        Check if the options are ok.

        Returns
        -------
        bool
            Whether the options are ok or not.
        """
        assert self.dataset in ['mobiliar', 'gvz'], "Invalid dataset"
        assert self.target_type in ['occurrence', 'damage_ratio'], "Invalid target type"
        assert self.random_state is None or isinstance(self.random_state, int), "Invalid random state"
        assert isinstance(self.use_event_attributes, bool), "Invalid use_event_attributes"
        assert isinstance(self.use_static_attributes, bool), "Invalid use_static_attributes"
        assert isinstance(self.use_all_static_attributes, bool), "Invalid use_all_static_attributes"

        return True

    @staticmethod
    def _parse_list_string(s: str) -> List[str]:
        """
        Parse a string like `['event', 'terrain', 'swf_map', 'flowacc', 'twi']`
        into a Python list of strings. Raises ValueError on invalid input.
        """
        if not s:
            return []
        try:
            val = ast.literal_eval(s)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Invalid list string: {e}") from e

        if not isinstance(val, list):
            raise ValueError("String does not represent a list")

        return [str(x) for x in val]
