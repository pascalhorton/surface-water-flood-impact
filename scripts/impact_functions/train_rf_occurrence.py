"""
Train a random forest model to predict the occurrence of damages.
"""
import time

from swafi.config import Config
from swafi.impact_rf import ImpactRandomForest
from swafi.impact_rf_options import ImpactRFOptions
from swafi.events import load_events_from_pickle

has_optuna = False
try:
    import optuna
    has_optuna = True
except ImportError:
    pass

OPTUNA_RANDOM = True
SAVE_MODEL = True
SHOW_PLOTS = False

config = Config()


def main():
    options = ImpactRFOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()

    # Load events
    events_filename = f'events_{options.dataset}_with_target_{options.event_file_label}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    if not options.optimize_with_optuna:
        rf = _setup_model(options, events)
        rf.fit()
        rf.assess_model_on_all_periods()
        rf.plot_feature_importance(tag='feature_importance_' + rf.options.run_name,
                                   dir_output=config.get('OUTPUT_DIR'))
        if SAVE_MODEL:
            rf.save_model(dir_output=config.get('OUTPUT_DIR'),
                          base_name='model_rf_' + rf.options.run_name)
            print(f"Model saved in {config.get('OUTPUT_DIR')}")

    else:
        rf = optimize_model_with_optuna(options, events, dir_plots=config.get('OUTPUT_DIR'))
        rf.assess_model_on_all_periods()
        rf.plot_feature_importance(tag='feature_importance_optuna_' + rf.options.run_name,
                                   dir_output=config.get('OUTPUT_DIR'))
        if SAVE_MODEL:
            rf.save_model(dir_output=config.get('OUTPUT_DIR'),
                          base_name='model_rf_optuna_' + rf.options.run_name)
            print(f"Model saved in {config.get('OUTPUT_DIR')}")


def _setup_model(options, events):
    rf = ImpactRandomForest(events, options=options)
    if rf.options.use_static_attributes or rf.options.use_event_attributes:
        rf.select_features(rf.options.replace_simple_features)
        rf.load_features(rf.options.simple_feature_classes)
    rf.split_sample()
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
        print("#" * 80)
        print(f"Trial {trial.number}")
        options_c = options.copy()
        options_c.generate_for_optuna(trial)
        options_c.print_options(show_optuna_params=True)
        rf_trial = _setup_model(options_c, events)

        start_time = time.time()

        # Fit the model
        rf_trial.fit()

        end_time = time.time()
        print(f"Model fitting took {end_time - start_time:.2f} seconds")

        # Assess the model
        score = rf_trial.compute_f1_score(rf_trial.x_valid, rf_trial.y_valid)

        return score

    study = _get_or_create_study(options, OPTUNA_RANDOM)
    study.optimize(optuna_objective, n_trials=options.optuna_trials_nb)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Fit the model with the best parameters
    options_best = options.copy()
    options_best.generate_for_optuna(best_trial)
    rf = _setup_model(options_best, events)
    rf.fit()

    return rf

def _get_or_create_study(options, random_sampler = False):
    file_path = f"./{options.optuna_study_name}.log"
    lock_obj = optuna.storages.journal.JournalFileOpenLock(file_path)  # For Windows
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(file_path, lock_obj=lock_obj)
    )

    sampler = None
    if random_sampler:
        sampler = optuna.samplers.RandomSampler()

    try:
        study = optuna.load_study(
            study_name=options.optuna_study_name,
            storage=storage,
            sampler=sampler
        )
        print(f"Study '{options.optuna_study_name}' already exists.")
    except KeyError:
        # If the study does not exist, create it
        study = optuna.create_study(
            study_name=options.optuna_study_name,
            storage=storage,
            direction="maximize",
            sampler=sampler
        )
        print(f"Study '{options.optuna_study_name}' created successfully.")

    return study

if __name__ == '__main__':
    main()
