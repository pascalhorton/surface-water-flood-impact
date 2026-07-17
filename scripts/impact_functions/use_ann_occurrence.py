"""
Test script for loading and evaluating a pre-trained ANN model.
"""
import logging
import keras
import random
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from swafi.config import Config
from swafi.impact_cnn import ImpactCnn
from swafi.utils.logging_setup import setup_logging
from swafi.utils.use_common import (
    assess, get_contracts_number, get_damages, get_damages_xr,
    get_events, create_prediction_dataset, ensure_precip_dataset,
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

    for i_x, x in enumerate(tqdm(xs, desc="Progress:", position=0)):
        for i_y, y in enumerate(ys):
            cell_id = domain.cids['ids_map'][i_y, i_x]
            if cell_id == 0:
                ds_pred['predict'][:, i_y, i_x] = np.nan
                continue

            exposure_cid = contracts_number[contracts_number['cid'] == cell_id]
            if len(exposure_cid) == 0 or exposure_cid['nb_contracts'].values[0] == 0:
                continue

            ds_pred['predict'][:, i_y, i_x] = 0

            x_input, _ = dg.get_batch_for_cid(cell_id)
            if len(x_input) == 0:
                continue

            event_dates = dg.get_event_dates_for_cid(cell_id)

            y_pred = ann.model.predict(x_input, verbose=0).squeeze()

            for i in range(len(event_dates)):
                if y_pred[i] == 0:
                    continue
                ref_date = pd.to_datetime(event_dates[i]).replace(hour=0, minute=0)
                ds_pred['predict'].loc[dict(time=ref_date, y=y, x=x)] = y_pred[i]

    ds_pred.to_netcdf(output_path)
    logger.info("Results saved to %s", output_path)
    ds_pred.close()

    if DO_ASSESS:
        assess(output_path, get_damages_xr(options.dataset, year_start, year_end),
               ignore_removed=True, relax_days=True, prob_threshold=THRESHOLD)


if __name__ == '__main__':
    main()
