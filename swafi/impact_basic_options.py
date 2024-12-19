"""
Class to define the options for the Transformer-based impact function.
"""
import argparse
import datetime
import copy


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
        print("-" * 80)
        self._print_basic_options()
        print("-" * 80)

    def _print_basic_options(self):
        """
        Print the options.
        """
        print(f"Options (run {self.run_name}):")
        print("- dataset: ", self.dataset)
        print("- event_file_label: ", self.event_file_label)
        print("- min_nb_claims: ", self.min_nb_claims)
        print("- target_type: ", self.target_type)
        print("- random_state: ", self.random_state)
        print("- use_event_attributes: ", self.use_event_attributes)
        print("- use_static_attributes: ", self.use_static_attributes)
        print("- use_all_static_attributes: ", self.use_all_static_attributes)

        if self.use_static_attributes or self.use_event_attributes:
            print("- simple_feature_classes: ", self.simple_feature_classes)
            print("- replace simple_features: ", self.replace_simple_features)

        if self.optimize_with_optuna:
            print("- optimize_with_optuna: ", self.optimize_with_optuna)
            print("- optuna_study_name: ", self.optuna_study_name)
            print("- optuna_trials_nb: ", self.optuna_trials_nb)
            print("- optuna_random_sampler: ", self.optuna_random_sampler)

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
