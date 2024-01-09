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
FACTOR_NEG_EVENTS = 1
WEIGHT_DENOMINATOR = 1

config = Config()


def main():
    parser = argparse.ArgumentParser(description="SWAFI DL")
    parser.add_argument("config", help="Configuration", type=int, default=0,
                        nargs='?')

    args = parser.parse_args()
    print("config: ", args.config)

    # Load events
    events_filename = f'events_{DATASET}_with_target_values_{LABEL_EVENT_FILE}.pickle'
    events = load_events_from_pickle(filename=events_filename)
    n_pos = events.count_positives()
    events.reduce_number_of_negatives(FACTOR_NEG_EVENTS * n_pos, random_state=42)

    # Create the impact function
    dl = ImpactDeepLearning(events, target_type='occurrence', random_state=42)

    # Configuration-specific changes
    if args.config == 0:  # Manual configuration
        pass
    elif args.config == 1:
        pass

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
    dl.load_features(['event', 'terrain', 'swf_map', 'flowacc', 'twi',
                      'land_cover', 'runoff_coeff'])

    dl.split_sample()
    dl.compute_balanced_class_weights()
    dl.compute_corrected_class_weights(weight_denominator=WEIGHT_DENOMINATOR)
    dl.fit()
    dl.assess_model_on_all_periods()


if __name__ == '__main__':
    main()
