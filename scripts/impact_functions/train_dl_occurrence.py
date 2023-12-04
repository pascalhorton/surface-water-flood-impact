"""
Train a deep learning model to predict the occurrence of damages.
"""

import argparse
import xarray as xr
from glob import glob

from swafi.config import Config
from swafi.impact_dl import ImpactDeepLearning
from swafi.events import load_events_from_pickle


LABEL_EVENT_FILE = 'original_w_prior_pluvial_occurrence'

config = Config()


def main():
    parser = argparse.ArgumentParser(description="SWAFI DL")
    parser.add_argument("config", help="Configuration", type=int, default=0,
                        nargs='?')

    args = parser.parse_args()
    print("config: ", args.config)

    # Load events
    events_filename = f'events_with_target_values_{LABEL_EVENT_FILE}.pickle'
    events = load_events_from_pickle(filename=events_filename)

    # Create the impact function
    dl = ImpactDeepLearning(events, target_type='occurrence', random_state=42)

    # Configuration-specific changes
    if args.config == 0:  # Manual configuration
        pass
    elif args.config == 1:
        pass

    # Load CombiPrecip files
    data_path = config.get('DIR_PRECIP')
    files = sorted(glob(f"{data_path}/*.nc"))
    precip = xr.open_mfdataset(files, parallel=True)
    dl.set_precipitation(precip)

    # Load DEM
    dem = xr.open_dataset(config.get('DEM_PATH'))
    dl.set_dem(dem)

    # Load static features
    dl.load_features(['event', 'terrain', 'swf_map', 'flowacc', 'twi',
                      'land_cover', 'runoff_coeff'])

    dl.split_sample()
    dl.compute_balanced_class_weights()
    dl.compute_corrected_class_weights(weight_denominator=27)
    dl.fit()
    dl.assess_model_on_all_periods()


if __name__ == '__main__':
    main()
