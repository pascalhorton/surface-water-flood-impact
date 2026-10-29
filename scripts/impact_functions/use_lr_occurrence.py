"""
Test script for loading and evaluating a pre-trained Logistic Regression model.
"""
import logging
import pickle
import numpy as np
from pathlib import Path

from swafi.config import Config
from swafi.impact_basic_options import ImpactBasicOptions
from swafi.impact_lr import ImpactLogisticRegression
from swafi.utils.logging_setup import setup_logging
from swafi.utils.event_extraction import detection_tag_in
from swafi.utils.use_common import (
    assess, get_contracts_number, get_damages, get_damages_xr,
    get_inference_events, create_prediction_dataset, GridPredictionWriter,
)

logger = logging.getLogger(__name__)

DO_ASSESS = True
MODEL = R"C:\Users\phorton\Documents\SWF\outputs\model_lr_.pkl"

# The event definition is read off the training catalogue named by the
# EVENTS_PATH_SIMPLE_* / EVENTS_PATH_CLASSIC config entry, which is the only
# record of how it was detected (see use_common.get_inference_events). Keeping
# a second copy of those settings here is what let inference drift from
# training. USE_TRAINING_REFERENCE ranks the '*_q' features against the
# training distribution, as the model was fitted; False re-estimates them on
# the test period.
USE_TRAINING_REFERENCE = True

config = Config()


def main():
    setup_logging(script_name='use_lr_occurrence')
    options = ImpactBasicOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()
    assert options.event_method in ['simple', 'classic'], "Invalid event method."

    payload = pickle.load(open(MODEL, "rb"))
    lr_model = payload['model']
    lr_mean = payload.get('mean', None)
    lr_std = payload.get('std', None)
    saved_features = payload.get('features', None)

    year_start = config.get('YEAR_START_TEST')
    year_end = config.get('YEAR_END_TEST')
    events = get_inference_events(options, year_start, year_end,
                                  use_training_reference=USE_TRAINING_REFERENCE)

    # The detection tag belongs in the NAME: the catalogue decides which
    # cell-days become candidates at all, and nothing inside a .nc records
    # which catalogue produced it, so without the tag a run against a second
    # catalogue silently reuses the first one's cached grid.
    catalogue = get_training_catalogue_path(options.event_method,
                                            options.precip_dataset)
    det_tag = detection_tag_in(catalogue) or '_detdefault'
    output_path = (
        Path(config.get('OUTPUT_DIR'))
        / f'pred_lr_{options.dataset}_{options.event_method}'
          f'_{options.precip_dataset}_{options.run_name}{det_tag}'
          f'_{year_start}-{year_end}.nc'
    )

    if output_path.exists():
        if DO_ASSESS:
            assess(output_path, get_damages_xr(options.dataset, year_start, year_end),
                   ignore_removed=True, relax_days=True, prob_threshold=0.5)
        return

    damages, _ = get_damages(options.dataset, year_start, year_end)
    contracts_number = get_contracts_number(damages)
    domain, xs, ys, ds_pred = create_prediction_dataset(year_start, year_end)

    lr = ImpactLogisticRegression(options)
    lr.model = lr_model
    lr.x_mean = lr_mean
    lr.x_std = lr_std
    # Align the default event features with the loaded events: sub-hourly
    # features (5-min dataset) are only included when present in the events.
    lr.update_potential_features(events.columns)
    lr.select_features(lr.options.replace_simple_features)
    features = lr.get_all_features(lr.options.simple_feature_classes)

    if saved_features is not None:
        assert lr.features == saved_features, (
            f"Feature mismatch between saved model and current options.\n"
            f"  Saved:   {saved_features}\n"
            f"  Current: {lr.features}"
        )

    writer = GridPredictionWriter(ds_pred, domain)
    writer.mask_outside_domain()

    # NaN masks, in the same precedence as the former per-cell loop:
    # cells without events stay at 0; cells with events but no features, then
    # no (or zero) exposure, are NaN.
    cids_events = set(events['cid'].unique()) & writer.get_map_cids()
    cids_features = set(features['cid'].unique())
    cids_exposure = set(
        contracts_number.loc[contracts_number['nb_contracts'] != 0, 'cid'])
    writer.fill_cells(cids_events - cids_features, np.nan)
    writer.fill_cells((cids_events & cids_features) - cids_exposure, np.nan)

    predict_cids = cids_events & cids_features & cids_exposure
    lr.set_events(events[events['cid'].isin(predict_cids)])
    lr.set_features(features)
    lr.set_exposure(contracts_number)
    lr.df.dropna(subset=lr.features, inplace=True)
    # Cells whose events were all dropped by the NaN filter are NaN too
    writer.fill_cells(predict_cids - set(lr.df['cid'].unique()), np.nan)

    if len(lr.df) > 0:
        x_input = lr.df[lr.features].to_numpy()
        if lr.x_mean is not None:
            x_input = (x_input - lr.x_mean) / lr.x_std
        y_pred = lr.model.predict_proba(x_input)[:, 1]
        writer.write_events(lr.df['cid'], lr.df['i_max_date'], y_pred)

    ds_pred.to_netcdf(output_path)
    logger.info("Results saved to %s", output_path)
    ds_pred.close()

    if DO_ASSESS:
        assess(output_path, get_damages_xr(options.dataset, year_start, year_end),
               ignore_removed=True, relax_days=True, prob_threshold=0.5)


if __name__ == '__main__':
    main()
