"""
Train a LightGBM model to predict the occurrence of damages.
"""
import logging
import time

from swafi.config import Config
from swafi.impact_lgbm import ImpactLGBM
from swafi.impact_lgbm_options import ImpactLGBMOptions
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

config = Config()


def main():
    setup_logging(script_name='train_lgbm_occurrence')
    options = ImpactLGBMOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()

    # Load events
    events_filename = f'events_{options.dataset}_with_target_{options.event_file_label}_{options.event_method}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    if not options.optimize_with_optuna:
        lgbm = _setup_model(options, events)
        lgbm.fit()
        lgbm.assess_model_on_all_periods(save_results=True,
                                         file_tag=f'lgbm_{lgbm.options.run_name}')
        lgbm.plot_feature_importance(tag='feature_importance_' + lgbm.options.run_name,
                                     dir_output=config.get('OUTPUT_DIR'))
        if SAVE_MODEL:
            lgbm.save_model(dir_output=config.get('OUTPUT_DIR'),
                            base_name=f'model_lgbm_{options.dataset}_{options.event_method}')
            logger.info("Model saved in %s", config.get('OUTPUT_DIR'))

    else:
        optimize_model_with_optuna(options, events)


def _setup_model(options, events):
    lgbm = ImpactLGBM(options, events)
    if lgbm.options.use_static_attributes or lgbm.options.use_event_attributes:
        lgbm.select_features(lgbm.options.replace_simple_features)
        lgbm.load_features(lgbm.options.simple_feature_classes)
    lgbm.split_sample(valid_test_size=0.25, test_size=0)
    lgbm.compute_balanced_class_weights()
    lgbm.compute_corrected_class_weights(
        weight_denominator=lgbm.options.weight_denominator)
    return lgbm


def optimize_model_with_optuna(options, events):
    """
    Optimize the model with Optuna.

    Parameters
    ----------
    options: ImpactLGBMOptions
        The options.
    events: pd.DataFrame
        The events.
    """
    if not has_optuna:
        raise ValueError("Optuna is not installed")

    def optuna_objective(trial):
        logger.info("%s", "#" * 80)
        logger.info("Trial %s", trial.number)
        options_c = options.copy()
        options_c.generate_for_optuna(trial)
        options_c.print_options(show_optuna_params=True)
        lgbm_trial = _setup_model(options_c, events)

        start_time = time.time()
        lgbm_trial.fit()
        end_time = time.time()
        logger.info("Model fitting took %.2f seconds", end_time - start_time)

        score = lgbm_trial.compute_f1_score(lgbm_trial.x_valid, lgbm_trial.y_valid)
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
