"""
Train a deep learning model to predict the occurrence of damages.
"""

import argparse
import xarray as xr
import rioxarray as rxr
from glob import glob

from swafi.config import Config
from swafi.impact_dl import ImpactDeepLearning
from swafi.events import load_events_from_pickle


DATASET = 'gvz'  # 'mobiliar' or 'gvz'
LABEL_EVENT_FILE = 'original_w_prior_pluvial_occurrence'
# FACTOR_NEG_EVENTS = 100
FACTOR_NEG_REDUCTION = 10
#WEIGHT_DENOMINATOR = 27
WEIGHT_DENOMINATOR = 5

config = Config()


def main():
    parser = argparse.ArgumentParser(description="SWAFI DL")
    parser.add_argument("config", type=int, default=-1, nargs='?',
                        help="Configuration ID (number corresponding to some options)")
    parser.add_argument("--do_not_use_precip", action="store_true",
                        help="Do not use precipitation data")
    parser.add_argument("--precip_resolution", type=int, default=1,
                        help="Desired resolution of the precipitation data [km]")
    parser.add_argument("--precip_time_step", type=int, default=6,
                        help="Desired time step of the precipitation data [hours]")
    parser.add_argument("--precip_days_before", type=int, default=4,
                        help="Number of days before the event to consider for the "
                             "precipitation data")
    parser.add_argument("--precip_days_after", type=int, default=2,
                        help="Number of days after the event to consider for the "
                             "precipitation data")

    args = parser.parse_args()
    print("config: ", args.config)

    # Main options
    use_precip = not args.do_not_use_precip
    precip_resolution = args.precip_resolution
    precip_time_step = args.precip_time_step
    precip_days_before = args.precip_days_before
    precip_days_after = args.precip_days_after

    # Load events
    events_filename = f'events_{DATASET}_with_target_values_{LABEL_EVENT_FILE}.pickle'
    events = load_events_from_pickle(filename=events_filename)
    n_pos = events.count_positives()
    # events.reduce_number_of_negatives(FACTOR_NEG_EVENTS * n_pos, random_state=42)

    # Configuration-specific changes
    interactive_mode = False
    if args.config == -1:  # Manual configuration
        interactive_mode = True
    elif args.config == 1:
        precip_time_step = 1
    elif args.config == 2:
        precip_time_step = 2
    elif args.config == 3:
        precip_time_step = 3
    elif args.config == 4:
        precip_time_step = 4
    elif args.config == 5:
        precip_time_step = 6
    elif args.config == 6:
        precip_time_step = 12
    elif args.config == 7:
        precip_time_step = 24

    # Create the impact function
    dl = ImpactDeepLearning(
        events, target_type='occurrence', random_state=42, precip_window_size=12,
        use_precip=use_precip, precip_resolution=precip_resolution,
        precip_time_step=precip_time_step, precip_days_before=precip_days_before,
        precip_days_after=precip_days_after)

    if use_precip:
        # Load DEM
        dem = rxr.open_rasterio(config.get('DEM_PATH'), masked=True).squeeze()
        dl.set_dem(dem)

        # Load CombiPrecip files
        data_path = config.get('DIR_PRECIP')
        files = sorted(glob(f"{data_path}/*.nc"))
        precip = xr.open_mfdataset(files, parallel=False)
        precip = precip.rename_vars({'CPC': 'precip'})
        precip = precip.rename({'REFERENCE_TS': 'time'})
        precip = precip.sel(x=dem.x, y=dem.y)  # Select the same domain as the DEM
        dl.set_precipitation(precip)

    # Load static features
    dl.load_features(['event', 'terrain', 'swf_map', 'flowacc', 'twi'])

    dl.split_sample()
    dl.reduce_negatives_for_training(FACTOR_NEG_REDUCTION)
    dl.compute_balanced_class_weights()
    dl.compute_corrected_class_weights(weight_denominator=WEIGHT_DENOMINATOR)
    dl.fit(dir_plots=config.get('OUTPUT_DIR'), show_plots=interactive_mode,
           tag=args.config)
    dl.assess_model_on_all_periods(show_plots=interactive_mode)


if __name__ == '__main__':
    main()
