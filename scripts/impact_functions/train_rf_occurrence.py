"""
Train a random forest model to predict the occurrence of damages.
"""
import logging
import time

from swafi.config import Config
from swafi.impact_rf import ImpactRandomForest
from swafi.impact_rf_options import ImpactRFOptions
from swafi.events import load_events_from_pickle
from swafi.utils.optuna import get_or_create_optuna_study
from swafi.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

has_optuna = False
try:
    import optuna
    has_optuna = True
except ImportError:
    pass

SAVE_MODEL = True
SHOW_PLOTS = False

config = Config()


def main():
    setup_logging(script_name='train_rf_occurrence')
    options = ImpactRFOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()

    # Load events
    events = load_events_from_pickle(filename=options.get_events_filename())
    events.check_precip_dataset(options.precip_dataset)

    if not options.optimize_with_optuna:
        rf = _setup_model(options, events)
        rf.fit()
        rf.tune_probability_threshold()
        rf.assess_model_on_all_periods(save_results=True, file_tag=f'rf_{rf.options.run_name}')
        rf.plot_feature_importance(tag='feature_importance_' + rf.options.run_name,
                                   dir_output=config.get('OUTPUT_DIR'))
        if SAVE_MODEL:
            rf.save_model(
                dir_output=config.get('OUTPUT_DIR'),
                base_name=f'model_rf_{options.dataset}_{options.event_method}'
                          f'_{options.precip_dataset}')
            logger.info("Model saved in %s", config.get('OUTPUT_DIR'))

    else:
        optimize_model_with_optuna(options, events, dir_plots=config.get('OUTPUT_DIR'))


def _setup_model(options, events):
    rf = ImpactRandomForest(options, events)
    if rf.options.use_static_attributes or rf.options.use_event_attributes:
        rf.select_features(rf.options.replace_simple_features)
        rf.load_features(rf.options.simple_feature_classes)
    rf.split_sample(valid_test_size=0.25, test_size=0)
    rf.compute_balanced_class_weights()
    rf.compute_corrected_class_weights(
        weight_denominator=rf.options.weight_denominator)
    return rf


def optimize_model_with_optuna(options, events, dir_plots=None):
    """
    Optimize the model with Optuna.

    Parameters
    ----------
    options: ImpactTransformerOptions
        The options.
    events: pd.DataFrame
        The events.
    dir_plots: str
        The directory where to save the plots.
    """
    if not has_optuna:
        raise ValueError("Optuna is not installed")

    def optuna_objective(trial):
        """
        The objective function for Optuna.

        Parameters
        ----------
        trial: optuna.Trial
            The trial.

        Returns
        -------
        float
            The score.
        """
        logger.info("%s", "#" * 80)
        logger.info("Trial %s", trial.number)
        options_c = options.copy()
        options_c.generate_for_optuna(trial)
        options_c.print_options(show_optuna_params=True)
        rf_trial = _setup_model(options_c, events)

        start_time = time.time()

        # Fit the model
        rf_trial.fit()

        end_time = time.time()
        logger.info("Model fitting took %.2f seconds", end_time - start_time)

        # Assess the model with a threshold-free metric (average precision),
        # so the hyperparameter search is not tied to a fixed decision threshold.
        score = rf_trial.compute_average_precision(
            rf_trial.x_valid, rf_trial.y_valid)

        return score

    study = get_or_create_optuna_study(options)
    study.optimize(optuna_objective, n_trials=options.optuna_trials_nb)

    logger.info("Number of finished trials: %s", len(study.trials))
    logger.info("Best trial:")
    best_trial = study.best_trial
    logger.info("  Value: %s", best_trial.value)
    logger.info("  Params: ")
    for key, value in best_trial.params.items():
        logger.info("    %s: %s", key, value)


if __name__ == '__main__':
    main()
