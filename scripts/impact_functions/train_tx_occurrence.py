"""
Train a deep learning model to predict the occurrence of damages.
"""

import logging
import random
import time
import keras
import numpy as np
import tensorflow as tf
import pandas as pd

from swafi.config import Config
from swafi.impact_tx import ImpactTransformer
from swafi.impact_tx_options import ImpactTransformerOptions
from swafi.events import load_events_from_pickle
from swafi.precip_combiprecip import CombiPrecip
from swafi.utils.optuna import get_or_create_optuna_study
from swafi.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

SAVE_MODEL = True
SHOW_PLOTS = False

config = Config()


def main():
    setup_logging(script_name='train_tx_occurrence')
    options = ImpactTransformerOptions()
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

    precip_hf = None
    precip_daily = None
    if options.use_precip:
        # Precipitation from the zarr store (config key PATH_PRECIP_HOURLY_ZARR)
        precip_hf = CombiPrecip(year_start, year_end)
        precip_daily = CombiPrecip(year_start, year_end)

    if not options.optimize_with_optuna:
        tx = _setup_model(options, events, precip_hf, precip_daily)
        tx.fit(dir_plots=config.get('OUTPUT_DIR'),
               tag=options.run_name, show_plots=SHOW_PLOTS)
        tx.assess_model_on_all_periods(save_results=True, file_tag=f'cnn_{tx.options.run_name}')
        if SAVE_MODEL:
            tx.save_model(dir_output=config.get('OUTPUT_DIR'), base_name='model_tx')
            logger.info("Model saved in %s", config.get('OUTPUT_DIR'))

    else:
        optimize_model_with_optuna(options, events, precip_hf, precip_daily,
                                   dir_plots=config.get('OUTPUT_DIR'))


def _setup_model(options, events, precip_hf, precip_daily):
    tx = ImpactTransformer(options, events)
    tx.set_precipitation_hf(precip_hf)
    tx.set_precipitation_daily(precip_daily)
    tx.remove_events_without_precipitation_data()
    tx.reduce_spatial_domain()
    if tx.options.use_static_attributes or tx.options.use_event_attributes:
        tx.select_features(tx.options.replace_simple_features)
        tx.load_features(tx.options.simple_feature_classes)
    tx.split_sample(valid_test_size=0.25, test_size=0)
    tx.reduce_negatives_for_training(tx.options.factor_neg_reduction)
    tx.compute_balanced_class_weights(tx.options.factor_neg_reduction)
    tx.compute_corrected_class_weights(
        weight_denominator=tx.options.weight_denominator)
    return tx


def optimize_model_with_optuna(options, events, precip_hf=None, precip_daily=None,
                               dir_plots=None):
    """
    Optimize the model with Optuna.

    Parameters
    ----------
    options: ImpactTransformerOptions
        The options.
    events: pd.DataFrame
        The events.
    precip_hf: CombiPrecip
        The high-frequency precipitation data.
    precip_daily: CombiPrecip
        The daily precipitation data.
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
        precip_hf.reset()
        precip_daily.reset()
        tx_trial = _setup_model(options_c, events, precip_hf, precip_daily)

        # Fit the model
        start_time = time.time()
        tx_trial.fit(do_plot=False)
        end_time = time.time()
        logger.info("Model fitting took %.2f seconds", end_time - start_time)

        # Assess the model
        score = tx_trial.compute_f1_score_full_data(tx_trial.dg_val)

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
