"""
Train a deep learning model to predict the occurrence of damages.
"""

import warnings
import xarray as xr
import rioxarray as rxr
import pandas as pd
from glob import glob

from swafi.config import Config
from swafi.impact_dl import ImpactDeepLearning, ImpactDeepLearningOptions
from swafi.events import load_events_from_pickle
from swafi.precip_combiprecip import CombiPrecip

has_optuna = False
try:
    import optuna
    has_optuna = True
    from optuna.storages import RDBStorage
except ImportError:
    pass

USE_SQLITE = True
DATASET = 'gvz'  # 'mobiliar' or 'gvz'
LABEL_EVENT_FILE = 'original_w_prior_pluvial_occurrence'

config = Config()

MISSING_DATES = CombiPrecip.missing
# Additional missing dates for ZH region (specific radar data)
MISSING_DATES.extend([
    ('2005-01-06', '2005-01-06'),
    ('2009-05-02', '2009-05-02'),
    ('2017-04-16', '2017-04-16'),
    ('2022-07-04', '2022-07-04'),
    ('2022-12-30', '2022-12-30')
])


def main():
    options = ImpactDeepLearningOptions()
    options.parse_args()
    # options.parser.print_help()
    options.print()
    assert options.is_ok()

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
        data_path = config.get('DIR_PRECIP')
        files = sorted(glob(f"{data_path}/*.nc"))
        precip = xr.open_mfdataset(files, parallel=False)
        precip = precip.rename_vars({'CPC': 'precip'})
        precip = precip.rename({'REFERENCE_TS': 'time'})
        if options.use_dem:
            precip = precip.sel(x=dem.x, y=dem.y)  # Select the same domain as the DEM

    if not options.optimize_with_optuna:
        dl = _setup_model(options, events, precip, dem)
        dl.fit(dir_plots=config.get('OUTPUT_DIR'),
               tag=options.run_name)
        dl.assess_model_on_all_periods()

    else:
        dl = optimize_model_with_optuna(options, events, precip, dem,
                                        dir_plots=config.get('OUTPUT_DIR'))
        dl.assess_model_on_all_periods()


def _setup_model(options, events, precip, dem):
    dl = ImpactDeepLearning(events, options=options)
    dl.set_dem(dem)
    dl.set_precipitation(precip)
    if dl.options.use_simple_features:
        dl.select_features(dl.options.simple_features)
        dl.load_features(dl.options.simple_feature_classes)
    dl.split_sample()
    dl.reduce_negatives_for_training(dl.options.factor_neg_reduction)
    dl.compute_balanced_class_weights()
    dl.compute_corrected_class_weights(
        weight_denominator=dl.options.weight_denominator)
    return dl


def optimize_model_with_optuna(options, events, precip=None, dem=None, dir_plots=None):
    """
    Optimize the model with Optuna.

    Parameters
    ----------
    options: ImpactDeepLearningOptions
        The options.
    events: pd.DataFrame
        The events.
    precip: xr.Dataset|None
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
            url=f'sqlite:///{options.optuna_study_name}.db'
        )

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
        dl_trial = _setup_model(options_c, events, precip, dem)

        # Fit the model
        dl_trial.fit(do_plot=False, silent=True)

        # Assess the model
        score = dl_trial.compute_f1_score(dl_trial.dg_val)

        return score

    if USE_SQLITE:
        study = optuna.load_study(
            study_name=options.optuna_study_name, storage=storage
        )
    else:
        study = optuna.create_study(
            study_name=options.optuna_study_name,
            direction='maximize'
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
    dl = _setup_model(options_best, events, precip, dem)
    dl.fit(dir_plots=dir_plots, tag='best_optuna_' + dl.options.run_name)

    return dl


if __name__ == '__main__':
    main()
