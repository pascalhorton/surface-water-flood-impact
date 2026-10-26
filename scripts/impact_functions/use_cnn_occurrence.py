"""
Test script for loading and evaluating a pre-trained CNN model.
"""
import logging
import keras
import random
import tensorflow as tf
import numpy as np
from pathlib import Path

from swafi.config import Config
from swafi.domain import Domain
from swafi.impact_cnn import ImpactCnn
from swafi.impact_cnn_options import ImpactCnnOptions
from swafi.impact_dl import WeightedBinaryCrossEntropy, CriticalSuccessIndex
from swafi.utils.logging_setup import setup_logging
from swafi.utils.use_common import (
    assess, get_contracts_number, get_damages, get_damages_xr,
    get_inference_events, resolve_precip_reference, filter_events_to_precip_domain,
    create_prediction_dataset, create_precipitation, ensure_precip_dataset,
    GridPredictionWriter, predict_events_in_chunks,
)

logger = logging.getLogger(__name__)

DO_ASSESS = True

# Full-domain scoring settings. These were False/False while every other
# use_*_occurrence.py script used True/True, which made the CNN's full-domain
# numbers quietly incomparable with the baselines' - the CNN was scored under
# the harsher rule.
#
# RELAX_DAYS moves a neighbouring day's prediction onto the claim day when that
# neighbour carries no claim of its own. It is justified: claim-date noise is
# measured, twice (9.6% of GVZ claims over 2005-2022, and half the unreachable
# cell-days on 2023-2024). But it is the GENEROUS bound - each move turns a
# false alarm and a miss into a hit and a correct negative, while the placebo
# puts only ~47% of +/-1 day associations down to a real storm. assess() now
# logs the ceiling under BOTH settings, so the bracket is always visible and
# this constant no longer hides half the answer.
#
# IGNORE_REMOVED excludes cell-days whose claim was dropped by the category
# selection. Those are not clean negatives: there WAS water damage, of another
# category, so scoring a prediction there as a false alarm charges the model
# for being roughly right.
RELAX_DAYS = True
IGNORE_REMOVED = True
MODEL = R"C:\Users\phorton\Documents\SWF\outputs\model_cnn_test_30.keras"
DATASET = 'mobiliar'  # 'mobiliar' or 'gvz'

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
    setup_logging(script_name='use_cnn_occurrence')
    cnn_model = keras.models.load_model(MODEL)

    options = cnn_model.options
    options.dataset = DATASET
    # Before print_options(): both it and is_ok() read the precipitation dataset.
    ensure_precip_dataset(options)
    options.print_options()
    assert options.is_ok()
    assert options.event_method in ['simple', 'classic'], "Invalid event method."

    keras.backend.clear_session()
    if options.random_state is not None:
        random.seed(options.random_state)
        np.random.seed(options.random_state)
        tf.random.set_seed(options.random_state)
        keras.utils.set_random_seed(options.random_state)

    if cnn_model.model is None:
        cnn_model.build_model()

    year_start = config.get('YEAR_START_TEST')
    year_end = config.get('YEAR_END_TEST')
    events = get_inference_events(options, year_start, year_end,
                                  use_training_reference=USE_TRAINING_REFERENCE)

    # Precipitation for CNN input, from the zarr store of the same dataset the
    # model was trained on.
    cpc = create_precipitation(options.precip_dataset, year_start, year_end)

    output_path = (
        Path(config.get('OUTPUT_DIR'))
        / f'pred_cnn_{options.dataset}_{options.event_method}'
          f'_{options.precip_dataset}_{options.run_name}_{year_start}-{year_end}.nc'
    )

    if output_path.exists():
        if DO_ASSESS:
            assess(output_path, get_damages_xr(options.dataset, year_start, year_end),
                   ignore_removed=IGNORE_REMOVED, relax_days=RELAX_DAYS,
                   prob_threshold=0.5)
        return

    damages, _ = get_damages(options.dataset, year_start, year_end)
    contracts_number = get_contracts_number(damages)
    domain, xs, ys, ds_pred = create_prediction_dataset(year_start, year_end)

    cnn = ImpactCnn(options)
    cnn.set_model(cnn_model)
    cnn.set_precipitation(cpc)
    features = None
    if cnn.options.use_static_attributes or cnn.options.use_event_attributes:
        # Align the default event features with the loaded events: sub-hourly
        # features (5-min dataset) are only included when present in the events.
        cnn.update_potential_features(events.columns)
        cnn.select_features(cnn.options.replace_simple_features)
        features = cnn.get_all_features(cnn.options.simple_feature_classes)

    # None when the statistics stored in the model are used; the precipitation
    # is then cropped to the domain the model was trained on, and so are the
    # events (no external reference to cover a wider one).
    precip_stats, cropped = resolve_precip_reference(
        cnn_model, options, cnn.precipitation_hf)
    if cropped:
        events = filter_events_to_precip_domain(
            events, cnn.precipitation_hf, options)
    dg = cnn.get_data_generator_inference(
        events=events,
        features=features,
        exposure=contracts_number,
        precip_stats=precip_stats,
    )
    if precip_stats is not None:
        precip_stats.close()

    writer = GridPredictionWriter(ds_pred, domain)
    writer.mask_outside_domain()

    # NaN masks, in the same precedence as the former per-cell loop: cells
    # without events stay at 0; cells with events but no (or zero) exposure
    # are NaN.
    cids_events = set(events['cid'].unique()) & writer.get_map_cids()
    cids_exposure = set(
        contracts_number.loc[contracts_number['nb_contracts'] != 0, 'cid'])
    writer.fill_cells(cids_events - cids_exposure, np.nan)

    predict_cids = cids_events & cids_exposure
    event_cids = dg.event_props[:, 3].astype(np.int64)
    idxs = np.where(np.isin(
        event_cids, np.fromiter(predict_cids, dtype=np.int64)))[0]
    if len(idxs) > 0:
        y_pred = predict_events_in_chunks(cnn.model, dg, idxs, chunk_size=1024)
        # Poisson head: rate -> P(>=1) = 1 - exp(-rate); no-op otherwise.
        y_pred = cnn._predictions_to_proba(y_pred)
        writer.write_events(event_cids[idxs], dg.event_props[idxs, 0], y_pred)

    ds_pred.to_netcdf(output_path)
    logger.info("Results saved to %s", output_path)
    ds_pred.close()

    if DO_ASSESS:
        assess(output_path, get_damages_xr(options.dataset, year_start, year_end),
               ignore_removed=IGNORE_REMOVED, relax_days=RELAX_DAYS,
               prob_threshold=0.5)


if __name__ == '__main__':
    main()
