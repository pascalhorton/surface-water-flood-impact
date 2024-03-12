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

    # Interactive mode (show plots)
    interactive_mode = False
    if options.run_id == 0:  # Manual configuration
        interactive_mode = True

    # Create the impact function
    dl = ImpactDeepLearning(events, options=options)

    if options.use_precip:
        if options.use_dem:
            # Load DEM
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)  # pyproj
                dem = rxr.open_rasterio(config.get('DEM_PATH'), masked=True).squeeze()
                dl.set_dem(dem)

        # Load CombiPrecip files
        data_path = config.get('DIR_PRECIP')
        files = sorted(glob(f"{data_path}/*.nc"))
        precip = xr.open_mfdataset(files, parallel=False)
        precip = precip.rename_vars({'CPC': 'precip'})
        precip = precip.rename({'REFERENCE_TS': 'time'})
        if options.use_dem:
            precip = precip.sel(x=dem.x, y=dem.y)  # Select the same domain as the DEM
        dl.set_precipitation(precip)

    # Load static features
    if options.use_simple_features:
        dl.select_features(options.simple_features)
        dl.load_features(options.simple_feature_classes)

    dl.split_sample()
    dl.reduce_negatives_for_training(options.factor_neg_reduction)
    dl.compute_balanced_class_weights()
    dl.compute_corrected_class_weights(weight_denominator=options.weight_denominator)
    if options.optimize_with_optuna:
        dl.optimize_model_with_optuna(n_jobs=options.optuna_jobs_nb,
                                      n_trials=options.optuna_trials_nb,
                                      dir_plots=config.get('OUTPUT_DIR'))
        dl.assess_model_on_all_periods()
    else:
        dl.fit(dir_plots=config.get('OUTPUT_DIR'),
               show_plots=interactive_mode,
               tag=options.run_id)
        dl.assess_model_on_all_periods()


if __name__ == '__main__':
    main()
