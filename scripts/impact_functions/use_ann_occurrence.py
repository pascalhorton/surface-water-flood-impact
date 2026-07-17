"""
Test script for loading and evaluating a pre-trained ANN model.
"""
import logging
import keras
import random
import tensorflow as tf
import numpy as np
from pathlib import Path

from swafi.config import Config
from swafi.impact_cnn import ImpactCnn
from swafi.utils.logging_setup import setup_logging
from swafi.utils.use_common import (
    assess, get_contracts_number, get_damages, get_damages_xr,
    get_events, create_prediction_dataset, ensure_precip_dataset,
    GridPredictionWriter, predict_events_in_chunks,
)

logger = logging.getLogger(__name__)

DO_ASSESS = True
DATASET = 'gvz'  # 'mobiliar' or 'gvz'
RUN_ID = '126'
MODEL = fR"C:\Users\phorton\Documents\SWF\outputs\model_ann_{DATASET}_{RUN_ID}.keras"
RUN_NAME = RUN_ID
EVENT_METHOD = 'simple'
THRESHOLD = 0.5

config = Config()


def main():
    setup_logging(script_name='use_ann_occurrence')
    ann_model = keras.models.load_model(MODEL)

    options = ann_model.options
    options.dataset = DATASET
    options.event_method = EVENT_METHOD
    options.run_name = RUN_NAME
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

    if ann_model.model is None:
        ann_model.build_model()

    year_start = config.get('YEAR_START_TEST')
    year_end = config.get('YEAR_END_TEST')
    events = get_events(year_start, year_end, options.event_method,
                        precip_dataset=options.precip_dataset)

    output_path = (
        Path(config.get('OUTPUT_DIR'))
        / f'pred_ann_{options.dataset}_{options.event_method}'
          f'_{options.precip_dataset}_{options.run_name}_{year_start}-{year_end}.nc'
    )

    if output_path.exists():
        if DO_ASSESS:
            assess(output_path, get_damages_xr(options.dataset, year_start, year_end),
                   ignore_removed=True, relax_days=True, prob_threshold=THRESHOLD)
        return

    damages, _ = get_damages(options.dataset, year_start, year_end)
    contracts_number = get_contracts_number(damages)
    domain, xs, ys, ds_pred = create_prediction_dataset(year_start, year_end,
                                                         fill_value=np.nan)

    ann = ImpactCnn(options)
    ann.set_model(ann_model)
    features = None
    if ann.options.use_static_attributes or ann.options.use_event_attributes:
        ann.select_features(ann.options.replace_simple_features)
        features = ann.get_all_features(ann.options.simple_feature_classes)

    dg = ann.get_data_generator_inference(
        events=events,
        features=features,
        exposure=contracts_number,
    )

    writer = GridPredictionWriter(ds_pred, domain)
    writer.mask_outside_domain()

    # Cells with nonzero exposure get a 0 background (the rest stays NaN, as
    # in the former per-cell loop), then their event predictions on top.
    cids_exposure = set(
        contracts_number.loc[contracts_number['nb_contracts'] != 0, 'cid'])
    predict_cids = writer.get_map_cids() & cids_exposure
    writer.fill_cells(predict_cids, 0.0)

    event_cids = dg.event_props[:, 3].astype(np.int64)
    idxs = np.where(np.isin(
        event_cids, np.fromiter(predict_cids, dtype=np.int64)))[0]
    if len(idxs) > 0:
        y_pred = predict_events_in_chunks(ann.model, dg, idxs, chunk_size=8192)
        writer.write_events(event_cids[idxs], dg.event_props[idxs, 0], y_pred)

    ds_pred.to_netcdf(output_path)
    logger.info("Results saved to %s", output_path)
    ds_pred.close()

    if DO_ASSESS:
        assess(output_path, get_damages_xr(options.dataset, year_start, year_end),
               ignore_removed=True, relax_days=True, prob_threshold=THRESHOLD)


if __name__ == '__main__':
    main()
