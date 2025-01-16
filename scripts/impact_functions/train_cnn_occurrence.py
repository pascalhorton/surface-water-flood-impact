"""
Train a CNN model to predict the occurrence of damages to buildings.
"""

import time
import warnings
import xarray as xr
import rioxarray as rxr
import pandas as pd

from swafi.config import Config
from swafi.impact_cnn import ImpactCnn
from swafi.impact_cnn_options import ImpactCnnOptions
from swafi.events import load_events_from_pickle
from swafi.precip_combiprecip import CombiPrecip
from swafi.utils.optuna import get_or_create_optuna_study

SAVE_MODEL = True
SHOW_PLOTS = False

config = Config()

MISSING_DATES = CombiPrecip.missing


def main():
    options = ImpactCnnOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()

    year_start = None
    year_end = None
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

    # Remove dates where the precipitation data is not available
    for date_range in MISSING_DATES:
        remove_start = (pd.to_datetime(date_range[0]) - pd.Timedelta(days=8))
        remove_end = (pd.to_datetime(date_range[1]) + pd.Timedelta(days=2))
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
                tag=options.run_name, show_plots=SHOW_PLOTS)
        cnn.assess_model_on_all_periods(save_results=True, file_tag=f'cnn_{cnn.options.run_name}')
        if SAVE_MODEL:
            cnn.save_model(dir_output=config.get('OUTPUT_DIR'), base_name='model_cnn')
            print(f"Model saved in {config.get('OUTPUT_DIR')}")

    else:
        optimize_model_with_optuna(options, events, precip, dem,
                                   dir_plots=config.get('OUTPUT_DIR'))


def _setup_model(options, events, precip, dem):
    cnn = ImpactCnn(events, options=options)
    cnn.set_dem(dem)
    cnn.set_precipitation(precip)
    cnn.remove_events_without_precipitation_data()
    cnn.reduce_spatial_domain(options.precip_window_size)
    if cnn.options.use_static_attributes or cnn.options.use_event_attributes:
        cnn.select_features(cnn.options.replace_simple_features)
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
        if precip is not None:
            precip.reset()
        cnn_trial = _setup_model(options_c, events, precip, dem)

        # Fit the model
        start_time = time.time()
        cnn_trial.fit(do_plot=False, silent=True)
        end_time = time.time()
        print(f"Model fitting took {end_time - start_time:.2f} seconds")

        # Assess the model
        score = cnn_trial.compute_f1_score_full_data(cnn_trial.dg_val)

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
