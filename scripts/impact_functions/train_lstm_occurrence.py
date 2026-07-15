"""
Train an LSTM+attention model to predict the occurrence of damages to buildings.
"""

import logging
import random
import time
import warnings
import keras
import numpy as np
import tensorflow as tf
import rioxarray as rxr

from swafi.config import Config
from swafi.impact_lstm import ImpactLstm
from swafi.impact_lstm_options import ImpactLstmOptions
from swafi.events import load_events_from_pickle
from swafi.precip_combiprecip import CombiPrecip
from swafi.utils.optuna import get_or_create_optuna_study
from swafi.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

SAVE_MODEL = True
SHOW_PLOTS = False

config = Config()


def main():
    setup_logging(script_name='train_lstm_occurrence')
    options = ImpactLstmOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()

    # Clear session and set the seed
    keras.backend.clear_session()
    if options.random_state is not None:
        random.seed(options.random_state)
        np.random.seed(options.random_state)
        tf.random.set_seed(options.random_state)
        keras.utils.set_random_seed(options.random_state)

    if options.dataset == 'mobiliar':
        year_start = config.get('YEAR_START_MOBILIAR')
        year_end = config.get('YEAR_END_MOBILIAR')
    elif options.dataset == 'gvz':
        year_start = config.get('YEAR_START_GVZ')
        year_end = config.get('YEAR_END_GVZ')
    else:
        raise ValueError(f'Dataset {options.dataset} not recognized.')

    # Load events
    events_filename = f'events_{options.dataset}_with_target_{options.event_file_label}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    dem = None
    precip = None
    if options.use_precip:
        if options.use_dem:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)  # pyproj
                dem = rxr.open_rasterio(config.get('DEM_PATH'), masked=True).squeeze()

        # Precipitation from the zarr store (config key PATH_PRECIP_HOURLY_ZARR)
        precip = CombiPrecip(year_start, year_end)

    if not options.optimize_with_optuna:
        lstm = _setup_model(options, events, precip, dem)
        lstm.fit(dir_plots=config.get('OUTPUT_DIR'),
                 tag=options.run_name, show_plots=SHOW_PLOTS)
        lstm.assess_model_on_all_periods(save_results=True,
                                         file_tag=f'lstm_{lstm.options.run_name}')
        if SAVE_MODEL:
            lstm.save_model(dir_output=config.get('OUTPUT_DIR'), base_name='model_lstm')
            logger.info("Model saved in %s", config.get('OUTPUT_DIR'))

    else:
        optimize_model_with_optuna(options, events, precip, dem,
                                   dir_plots=config.get('OUTPUT_DIR'))


def _setup_model(options, events, precip, dem):
    lstm = ImpactLstm(options, events)
    lstm.set_dem(dem)
    lstm.set_precipitation(precip)
    lstm.remove_events_without_precipitation_data()
    if lstm.options.use_static_attributes or lstm.options.use_event_attributes:
        lstm.select_features(lstm.options.replace_simple_features)
        lstm.load_features(lstm.options.simple_feature_classes)
    lstm.split_sample(valid_test_size=0.25, test_size=0)
    lstm.reduce_negatives_for_training(lstm.options.factor_neg_reduction)
    lstm.compute_balanced_class_weights(lstm.options.factor_neg_reduction)
    lstm.compute_corrected_class_weights(
        weight_denominator=lstm.options.weight_denominator)
    return lstm


def optimize_model_with_optuna(options, events, precip=None, dem=None,
                                dir_plots=None):
    """
    Optimize the model with Optuna.

    Parameters
    ----------
    options: ImpactLstmOptions
        The options.
    events: pd.DataFrame
        The events.
    precip: Precipitation|None
        The precipitation data.
    dem: xr.Dataset|None
        The DEM data.
    dir_plots: str
        The directory where to save the plots.
    """

    def optuna_objective(trial):
        logger.info("%s", "#" * 80)
        logger.info("Trial %s", trial.number)
        options_c = options.copy()
        options_c.generate_for_optuna(trial)
        options_c.print_options(show_optuna_params=True)
        if precip is not None:
            precip.reset()
        lstm_trial = _setup_model(options_c, events, precip, dem)

        start_time = time.time()
        lstm_trial.fit(do_plot=False)
        end_time = time.time()
        logger.info("Model fitting took %.2f seconds", end_time - start_time)

        score = lstm_trial.compute_f1_score_full_data(lstm_trial.dg_val)
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
