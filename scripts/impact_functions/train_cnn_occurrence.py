"""
Train a deep learning model to predict the occurrence of damages.
"""

import warnings
import xarray as xr
import rioxarray as rxr
import pandas as pd
from glob import glob

from swafi.config import Config
from swafi.impact_cnn import ImpactCnn, ImpactCnnOptions
from swafi.events import load_events_from_pickle
from swafi.precip_combiprecip import CombiPrecip

has_optuna = False
try:
    import optuna

    has_optuna = True
    from optuna.storages import RDBStorage, JournalStorage, JournalFileStorage
except ImportError:
    pass

USE_SQLITE = False
USE_TXTFILE = True
OPTUNA_RANDOM = True
DATASET = 'mobiliar'  # 'mobiliar' or 'gvz'
LABEL_EVENT_FILE = 'original_w_prior_pluvial_occurrence'
SAVE_MODEL = True

config = Config()

MISSING_DATES = CombiPrecip.missing


def main():
    options = ImpactCnnOptions()
    options.parse_args()
    # options.parser.print_help()
    options.print()
    assert options.is_ok()

    year_start = None
    year_end = None
    if DATASET == 'mobiliar':
        year_start = config.get('YEAR_START_MOBILIAR')
        year_end = config.get('YEAR_END_MOBILIAR')
    elif DATASET == 'gvz':
        year_start = config.get('YEAR_START_GVZ')
        year_end = config.get('YEAR_END_GVZ')
    else:
        raise ValueError(f'Dataset {DATASET} not recognized.')

    # Load events
    events_filename = f'events_{DATASET}_with_target_values_{LABEL_EVENT_FILE}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    # Remove dates where the precipitation data is not available
    for date_range in MISSING_DATES:
        remove_start = (pd.to_datetime(date_range[0])
                        - pd.Timedelta(days=options.precip_days_before + 1))
        remove_end = (pd.to_datetime(date_range[1])
                      + pd.Timedelta(days=options.precip_days_after + 1))
        events.remove_period(remove_start, remove_end)

    dem = None
    precip = None
    if options.use_precip:
        if options.use_dem:
            # Load DEM
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)  # pyproj
                dem = rxr.open_rasterio(config.get('DEM_PATH'), masked=True).squeeze()

        # Load CombiPrecip files
        precip = CombiPrecip(year_start, year_end)
        precip.set_data_path(config.get('DIR_PRECIP'))

    if not options.optimize_with_optuna:
        cnn = _setup_model(options, events, precip, dem)
        cnn.fit(dir_plots=config.get('OUTPUT_DIR'),
                tag=options.run_name, show_plots=True)
        cnn.assess_model_on_all_periods()
        if SAVE_MODEL:
            cnn.save_model(dir_output=config.get('OUTPUT_DIR'))
            print(f"Model saved in {config.get('OUTPUT_DIR')}")

    else:
        cnn = optimize_model_with_optuna(options, events, precip, dem,
                                         dir_plots=config.get('OUTPUT_DIR'))
        cnn.assess_model_on_all_periods()


def _setup_model(options, events, precip, dem):
    cnn = ImpactCnn(events, options=options)
    cnn.set_dem(dem)
    cnn.set_precipitation(precip)
    cnn.remove_events_without_precipitation_data()
    cnn.reduce_spatial_domain(options.precip_window_size)
    if cnn.options.use_simple_features:
        cnn.select_features(cnn.options.simple_features)
        cnn.load_features(cnn.options.simple_feature_classes)
    cnn.split_sample()
    cnn.reduce_negatives_for_training(cnn.options.factor_neg_reduction)
    cnn.compute_balanced_class_weights()
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
    if not has_optuna:
        raise ValueError("Optuna is not installed")

    if USE_SQLITE:
        storage = RDBStorage(
            url=f'sqlite:///{options.optuna_study_name}.db',
            engine_kwargs={"connect_args": {"timeout": 60.0}}
        )
    elif USE_TXTFILE:
        storage = JournalStorage(
            JournalFileStorage(f"{options.optuna_study_name}.log")
        )
    else:
        raise ValueError("No storage specified")

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
        options_c = options.copy()
        options_c.generate_for_optuna(trial)
        cnn_trial = _setup_model(options_c, events, precip, dem)

        # Fit the model
        cnn_trial.fit(do_plot=False, silent=True)

        # Assess the model
        score = cnn_trial.compute_f1_score(cnn_trial.dg_val)

        return score

    sampler = None
    if OPTUNA_RANDOM:
        sampler = optuna.samplers.RandomSampler()

    if USE_SQLITE or USE_TXTFILE:
        study = optuna.load_study(
            study_name=options.optuna_study_name,
            storage=storage,
            sampler=sampler
        )
    else:
        study = optuna.create_study(
            study_name=options.optuna_study_name,
            direction='maximize',
            sampler=sampler
        )
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
    cnn = _setup_model(options_best, events, precip, dem)
    cnn.fit(dir_plots=dir_plots, tag='best_optuna_' + cnn.options.run_name)

    return cnn


if __name__ == '__main__':
    main()
