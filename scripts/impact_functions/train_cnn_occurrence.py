"""
Train a CNN model to predict the occurrence of damages to buildings.
"""

import logging
import random
import time
import warnings
import keras
import numpy as np
import tensorflow as tf
import xarray as xr
import rioxarray as rxr
import pandas as pd

from swafi.config import Config
from swafi.impact_cnn import ImpactCnn
from swafi.impact_cnn_options import ImpactCnnOptions
from swafi.events import load_events_from_pickle
from swafi.precip_combiprecip import CombiPrecip
from swafi.utils.optuna import get_or_create_optuna_study
from swafi.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

SAVE_MODEL = True
SHOW_PLOTS = False

config = Config()


def main():
    setup_logging(script_name='train_cnn_occurrence')
    options = ImpactCnnOptions()
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
    events = load_events_from_pickle(filename=options.get_events_filename())
    events.check_precip_dataset(options.precip_dataset)

    dem = None
    precip = None
    if options.use_precip:
        if options.use_dem:
            # Load DEM
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)  # pyproj
                dem = rxr.open_rasterio(config.get('DEM_PATH'), masked=True).squeeze()

        # Precipitation from the zarr store (config key PATH_PRECIP_HOURLY_ZARR)
        precip = CombiPrecip(year_start, year_end)

    if not options.optimize_with_optuna:
        cnn = _setup_model(options, events, precip, dem)
        cnn.fit(dir_plots=config.get('OUTPUT_DIR'),
                tag=options.run_name, show_plots=SHOW_PLOTS)
        cnn.assess_model_on_all_periods(save_results=True, file_tag=f'cnn_{cnn.options.run_name}')
        if SAVE_MODEL:
            cnn.save_model(dir_output=config.get('OUTPUT_DIR'), base_name='model_cnn')
            logger.info("Model saved in %s", config.get('OUTPUT_DIR'))

    else:
        optimize_model_with_optuna(options, events, precip, dem,
                                   dir_plots=config.get('OUTPUT_DIR'))


def _setup_model(options, events, precip, dem):
    cnn = ImpactCnn(options, events)
    cnn.set_dem(dem)
    cnn.set_precipitation(precip)
    cnn.remove_events_without_precipitation_data()
    cnn.reduce_spatial_domain(options.precip_window_size)
    if cnn.options.use_static_attributes or cnn.options.use_event_attributes:
        cnn.select_features(cnn.options.replace_simple_features)
        cnn.load_features(cnn.options.simple_feature_classes)
    cnn.split_sample(valid_test_size=0.25, test_size=0)
    cnn.reduce_negatives_for_training(cnn.options.factor_neg_reduction)
    if not options.use_poisson_head:
        cnn.compute_balanced_class_weights(cnn.options.factor_neg_reduction)
        cnn.compute_corrected_class_weights(
            weight_denominator=cnn.options.weight_denominator)
    return cnn


def optimize_model_with_optuna(options, events, precip=None, dem=None, dir_plots=None):
    """
    Optimize the model with Optuna.

    Parameters
    ----------
    options: ImpactCnnOptions
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
        if precip is not None:
            precip.reset()
        cnn_trial = _setup_model(options_c, events, precip, dem)

        # Fit the model
        start_time = time.time()
        cnn_trial.fit(do_plot=False)
        end_time = time.time()
        logger.info("Model fitting took %.2f seconds", end_time - start_time)

        # Assess the model
        score = cnn_trial.compute_f1_score_full_data(cnn_trial.dg_val)

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
