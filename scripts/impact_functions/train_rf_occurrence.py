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
        threshold = rf.tune_probability_threshold()
        logger.info("Optimal probability threshold (tuned on validation): %.4f",
                    threshold)
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

    if options.optuna_save_best:
        save_best_model(options, events, study, dir_output=dir_plots)


def save_best_model(options, events, study, dir_output):
    """
    Refit the best trial of the study and save two models:

    1. the split model, fit on the chronological training split and assessed on
       the held-out validation split (the honest performance estimate), with
       its decision threshold tuned on that validation split;
    2. the full-period model, refit with the same hyperparameters on the whole
       period (training + validation) for deployment, reusing the threshold of
       the split model (no held-out data remains to tune it on).

    The best trial only stores the tuned hyperparameters; the fixed ones
    (n_estimators, weight_denominator, ...) are taken from the base options.
    With the fixed random state and the deterministic (chronological) split,
    refitting reproduces the trial's model exactly. The decision threshold was
    not tuned during the search (the objective is the threshold-free average
    precision), so it is tuned here.

    Parameters
    ----------
    options: ImpactRFOptions
        The base options (carrying the fixed hyperparameters).
    events: Events
        The events object.
    study: optuna.study.Study
        The completed study.
    dir_output: str
        The directory where to save the models, results and plots.
    """
    best_trial = study.best_trial
    logger.info("Refitting the best model (trial %s, value %.5f) to save it.",
                best_trial.number, best_trial.value)

    options_best = options.copy()
    # The trial params share the option names, so apply them by name; the
    # hyperparameters kept out of the search retain their base values.
    for key, value in best_trial.params.items():
        setattr(options_best, key, value)
    # Behave as a plain fit from here on (show the resolved params, not the
    # Optuna search space).
    options_best.optimize_with_optuna = False
    options_best.print_options(show_optuna_params=True)

    base_name = (f'model_rf_{options_best.dataset}_{options_best.event_method}'
                 f'_{options_best.precip_dataset}')

    # 1. Split model: fit on the training split, assess and tune the threshold
    # on the held-out validation split.
    rf = _setup_model(options_best, events)
    rf.fit()
    threshold = rf.tune_probability_threshold()
    logger.info("Optimal probability threshold (tuned on validation): %.4f",
                threshold)
    rf.assess_model_on_all_periods(save_results=True,
                                   file_tag=f'rf_{options_best.run_name}')
    rf.plot_feature_importance(
        tag='feature_importance_' + options_best.run_name, dir_output=dir_output)
    rf.save_model(dir_output=dir_output, base_name=base_name)
    logger.info("Split model saved in %s", dir_output)

    # 2. Full-period model: refit the same hyperparameters on the whole period
    # for deployment, keeping the threshold selected on the validation split
    # (no held-out data remains to tune or assess it on).
    logger.info("Refitting the best model on the whole period (deployment model).")
    rf.merge_valid_test_into_train()
    rf.compute_balanced_class_weights()
    rf.compute_corrected_class_weights(
        weight_denominator=options_best.weight_denominator)
    rf.fit()
    rf.probability_threshold = threshold
    rf.plot_feature_importance(
        tag='feature_importance_full_' + options_best.run_name,
        dir_output=dir_output)
    rf.save_model(dir_output=dir_output, base_name=base_name + '_full')
    logger.info("Full-period model saved in %s (threshold %.4f from the split "
                "model).", dir_output, threshold)


if __name__ == '__main__':
    main()
