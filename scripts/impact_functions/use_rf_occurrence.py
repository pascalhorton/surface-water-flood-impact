"""
Test script for loading and evaluating a pre-trained Random Forest model.
"""
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from swafi.config import Config
from swafi.impact_rf_options import ImpactRFOptions
from swafi.impact_rf import ImpactRandomForest
from swafi.utils.logging_setup import setup_logging
from swafi.utils.use_common import (
    assess, get_contracts_number, get_damages, get_damages_xr,
    get_events, create_prediction_dataset,
)

logger = logging.getLogger(__name__)

DO_ASSESS = True
MODEL = R"C:\Users\phorton\Documents\SWF\outputs\model_rf_2025-08-29_160926.pkl"

config = Config()


def main():
    setup_logging(script_name='use_rf_occurrence')
    options = ImpactRFOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()
    assert options.event_method in ['simple', 'classic'], "Invalid event method."

    rf_model = pickle.load(open(MODEL, "rb"))

    year_start = config.get('YEAR_START_TEST')
    year_end = config.get('YEAR_END_TEST')
    events = get_events(year_start, year_end, options.event_method,
                        precip_dataset=options.precip_dataset)

    output_path = (
        Path(config.get('OUTPUT_DIR'))
        / f'pred_rf_{options.dataset}_{options.event_method}'
          f'_{options.precip_dataset}_{year_start}-{year_end}.nc'
    )

    if output_path.exists():
        if DO_ASSESS:
            assess(output_path, get_damages_xr(options.dataset, year_start, year_end),
                   ignore_removed=True, relax_days=True, prob_threshold=0.5)
        return

    damages, _ = get_damages(options.dataset, year_start, year_end)
    contracts_number = get_contracts_number(damages)
    domain, xs, ys, ds_pred = create_prediction_dataset(year_start, year_end)

    rf = ImpactRandomForest(options)
    rf.set_model(rf_model)
    rf.select_features(rf.options.replace_simple_features)
    features = rf.get_all_features(rf.options.simple_feature_classes)

    for i_x, x in enumerate(tqdm(xs, desc="Progress:", position=0)):
        for i_y, y in enumerate(ys):
            cell_id = domain.cids['ids_map'][i_y, i_x]
            if cell_id == 0:
                ds_pred['predict'][:, i_y, i_x] = np.nan
                continue

            exposure_cid = contracts_number[contracts_number['cid'] == cell_id]
            if len(exposure_cid) == 0 or exposure_cid['nb_contracts'].values[0] == 0:
                ds_pred['predict'][:, i_y, i_x] = np.nan
                continue

            cell_events = events[events['cid'] == cell_id]
            if len(cell_events) == 0:
                continue

            features_cid = features[features['cid'] == cell_id]
            if len(features_cid) == 0:
                ds_pred['predict'][:, i_y, i_x] = np.nan
                continue

            rf.set_events(cell_events)
            rf.set_features(features_cid)
            rf.set_exposure(exposure_cid)
            rf.df.dropna(subset=rf.features, inplace=True)
            x_input = rf.df[rf.features].to_numpy()
            if len(x_input) == 0:
                ds_pred['predict'][:, i_y, i_x] = np.nan
                continue

            y_pred = rf.model.predict_proba(x_input)[:, 1]  # Probability of class 1
            assert len(y_pred) == len(rf.df)

            # Loop over events and store the target value at the correct date
            for i, (_, row) in enumerate(rf.df.iterrows()):
                if y_pred[i] == 0:
                    continue
                # Store the prediction at the date of the event
                ref_date = pd.to_datetime(row['i_max_date']).replace(hour=0, minute=0)
                ds_pred['predict'].loc[dict(time=ref_date, y=y, x=x)] = y_pred[i]

    ds_pred.to_netcdf(output_path)
    logger.info("Results saved to %s", output_path)
    ds_pred.close()

    if DO_ASSESS:
        assess(output_path, get_damages_xr(options.dataset, year_start, year_end),
               ignore_removed=True, relax_days=True, prob_threshold=0.5)


if __name__ == '__main__':
    main()
