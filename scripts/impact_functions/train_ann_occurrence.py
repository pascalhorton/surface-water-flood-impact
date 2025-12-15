"""
Train an ANN model to predict the occurrence of damages to buildings.
"""

import random
import time
import keras
import numpy as np
import tensorflow as tf
import pandas as pd

from swafi.config import Config
from swafi.impact_cnn import ImpactCnn
from swafi.impact_cnn_options import ImpactCnnOptions
from swafi.events import load_events_from_pickle
from swafi.utils.optuna import get_or_create_optuna_study

SAVE_MODEL = True
SHOW_PLOTS = False

config = Config()


def main():
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

    # Load events
    events_filename = f'events_{options.dataset}_with_target_{options.event_file_label}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    if not options.optimize_with_optuna:
        ann = _setup_model(options, events)
        ann.fit(
            dir_plots=config.get('OUTPUT_DIR'),
            tag=options.run_name,
            show_plots=SHOW_PLOTS
        )
        ann.assess_model_on_all_periods(save_results=True, file_tag=f'ann_{ann.options.run_name}')
        if SAVE_MODEL:
            ann.save_model(dir_output=config.get('OUTPUT_DIR'), base_name='model_cnn')
            print(f"Model saved in {config.get('OUTPUT_DIR')}")

    else:
        optimize_model_with_optuna(options, events)


def _setup_model(options, events):
    ann = ImpactCnn(options, events)
    ann.select_features(ann.options.replace_simple_features)
    ann.load_features(ann.options.simple_feature_classes)
    ann.split_sample()
    ann.reduce_negatives_for_training(ann.options.factor_neg_reduction)
    ann.compute_balanced_class_weights(ann.options.factor_neg_reduction)
    ann.compute_corrected_class_weights(
        weight_denominator=ann.options.weight_denominator)
    return ann


def optimize_model_with_optuna(options, events):
    """
    Optimize the model with Optuna.

    Parameters
    ----------
    options: ImpactCnnOptions
        The options.
    events: pd.DataFrame
        The events.
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
        print("#" * 80)
        print(f"Trial {trial.number}")
        options_c = options.copy()
        options_c.generate_for_optuna(trial)
        options_c.print_options(show_optuna_params=True)
        ann_trial = _setup_model(options_c, events)

        # Fit the model
        start_time = time.time()
        ann_trial.fit(do_plot=False)
        end_time = time.time()
        print(f"Model fitting took {end_time - start_time:.2f} seconds")

        # Assess the model
        score = ann_trial.compute_f1_score_full_data(ann_trial.dg_val)

        return score

    study = get_or_create_optuna_study(options)
    study.optimize(optuna_objective, n_trials=options.optuna_trials_nb)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == '__main__':
    main()
