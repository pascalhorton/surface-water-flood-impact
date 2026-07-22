"""
Test script for loading and evaluating a pre-trained Random Forest model.
"""
import logging
import pickle
import numpy as np
from pathlib import Path

from swafi.config import Config
from swafi.impact_rf_options import ImpactRFOptions
from swafi.impact_rf import ImpactRandomForest
from swafi.utils.logging_setup import setup_logging
from swafi.utils.use_common import (
    assess, get_contracts_number, get_damages, get_damages_xr,
    get_events, create_prediction_dataset, GridPredictionWriter,
)

logger = logging.getLogger(__name__)

DO_ASSESS = True
SWEEP_THRESHOLDS = True  # also log scores across thresholds and the best operating point
MODEL = R"C:\Users\phorton\Documents\SWF\outputs\model_rf_2025-08-29_160926.pkl"
# Per-cell training reference (the '*_ref.pkl' saved by extract_precipitation_events
# for the simple method). When set, the test events are normalised against the
# training distribution instead of the test period. None keeps the old behaviour.
REFERENCE_PATH = None
# Event detection settings: must match the ones used to extract the training
# events (extract_precipitation_events.py), otherwise the test events follow
# another event definition than the model was trained on.
DETECTION_WINDOW_H = 1
DETECTION_THRESHOLD = None  # e.g. 10 with DETECTION_WINDOW_H = 12 for p_12h >= 10mm
DETECTION_CENTERED = True  # centre the detection window on the step it labels
DETECTION_PEAK_DAYS = True  # date the events on each exceeding window's peak

config = Config()


def main():
    setup_logging(script_name='use_rf_occurrence')
    options = ImpactRFOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()
    assert options.event_method in ['simple', 'classic'], "Invalid event method."

    payload = pickle.load(open(MODEL, "rb"))
    if isinstance(payload, dict):
        rf_model = payload['model']
        saved_features = payload.get('features', None)
        prob_threshold = payload.get('probability_threshold', 0.5)
    else:  # Legacy format: bare sklearn model, without the feature list
        rf_model = payload
        saved_features = None
        prob_threshold = 0.5

    year_start = config.get('YEAR_START_TEST')
    year_end = config.get('YEAR_END_TEST')
    events = get_events(year_start, year_end, options.event_method,
                        precip_dataset=options.precip_dataset,
                        detection_window_h=DETECTION_WINDOW_H,
                        detection_threshold=DETECTION_THRESHOLD,
                        detection_centered=DETECTION_CENTERED,
                        detection_peak_days=DETECTION_PEAK_DAYS,
                        reference_path=REFERENCE_PATH)

    output_path = (
        Path(config.get('OUTPUT_DIR'))
        / f'pred_rf_{options.dataset}_{options.event_method}'
          f'_{options.precip_dataset}_{year_start}-{year_end}.nc'
    )

    damages, _ = get_damages(options.dataset, year_start, year_end)
    contracts_number = get_contracts_number(damages)
    domain, xs, ys, ds_pred = create_prediction_dataset(year_start, year_end)

    rf = ImpactRandomForest(options)
    rf.set_model(rf_model)
    # Align the default event features with the loaded events: sub-hourly
    # features (5-min dataset) are only included when present in the events.
    rf.update_potential_features(events.columns)
    rf.select_features(rf.options.replace_simple_features)
    features = rf.get_all_features(rf.options.simple_feature_classes)

    if saved_features is not None:
        assert rf.features == saved_features, (
            f"Feature mismatch between saved model and current options.\n"
            f"  Saved:   {saved_features}\n"
            f"  Current: {rf.features}"
        )

    # Legacy models carry no feature list: at least check the feature count
    n_features_model = getattr(rf_model, 'n_features_in_', None)
    if n_features_model is not None and n_features_model != len(rf.features):
        raise ValueError(
            f"The loaded model expects {n_features_model} features, but the "
            f"current setup provides {len(rf.features)}: {rf.features}. Check "
            f"that the precipitation dataset and feature options match the "
            f"ones used for training.")

    writer = GridPredictionWriter(ds_pred, domain)
    writer.mask_outside_domain()

    # NaN masks, in the same precedence as the former per-cell loop: cells
    # without (or zero) exposure are NaN; remaining cells without events stay
    # at 0; cells with events but no features are NaN.
    map_cids = writer.get_map_cids()
    cids_events = set(events['cid'].unique()) & map_cids
    cids_features = set(features['cid'].unique())
    cids_exposure = set(
        contracts_number.loc[contracts_number['nb_contracts'] != 0, 'cid'])
    writer.fill_cells(map_cids - cids_exposure, np.nan)
    writer.fill_cells((cids_events & cids_exposure) - cids_features, np.nan)

    predict_cids = cids_events & cids_exposure & cids_features
    rf.set_events(events[events['cid'].isin(predict_cids)])
    rf.set_features(features)
    rf.set_exposure(contracts_number)
    rf.df.dropna(subset=rf.features, inplace=True)
    # Cells whose events were all dropped by the NaN filter are NaN too
    writer.fill_cells(predict_cids - set(rf.df['cid'].unique()), np.nan)

    if len(rf.df) > 0:
        x_input = rf.df[rf.features].to_numpy()
        y_pred = rf.model.predict_proba(x_input)[:, 1]  # Probability of class 1
        writer.write_events(rf.df['cid'], rf.df['i_max_date'], y_pred)

    ds_pred.to_netcdf(output_path)
    logger.info("Results saved to %s", output_path)
    ds_pred.close()

    if DO_ASSESS:
        assess(output_path, get_damages_xr(options.dataset, year_start, year_end),
               ignore_removed=True, relax_days=True, prob_threshold=prob_threshold,
               sweep_thresholds=SWEEP_THRESHOLDS)


if __name__ == '__main__':
    main()
