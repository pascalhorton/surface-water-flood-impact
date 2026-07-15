"""
Test script for loading and evaluating a pre-trained CNN model.
"""
import logging
import keras
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm

from swafi.config import Config
from swafi.domain import Domain
from swafi.impact_cnn import ImpactCnn
from swafi.impact_cnn_options import ImpactCnnOptions
from swafi.impact_dl import WeightedBinaryCrossEntropy, CriticalSuccessIndex
from swafi.precip_combiprecip import CombiPrecip
from swafi.utils.logging_setup import setup_logging
from swafi.utils.use_common import (
    assess, get_contracts_number, get_damages, get_damages_xr,
    get_events, create_prediction_dataset,
)

logger = logging.getLogger(__name__)

DO_ASSESS = True
MODEL = R"C:\Users\phorton\Documents\SWF\outputs\model_cnn_test_30.keras"
PRECIP_STATS_PATH = R"C:\Users\phorton\Documents\SWF\data\cpc_statistics_2005-2022.nc"
DATASET = 'mobiliar'  # 'mobiliar' or 'gvz'

config = Config()


def main():
    setup_logging(script_name='use_cnn_occurrence')
    cnn_model = keras.models.load_model(MODEL)

    options = cnn_model.options
    options.dataset = DATASET
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
    events = get_events(year_start, year_end, options.event_method)

    # Precipitation for CNN input, from the zarr store (PATH_PRECIP_HOURLY_ZARR)
    cpc = CombiPrecip(year_start, year_end)

    output_path = (
        Path(config.get('OUTPUT_DIR'))
        / f'pred_cnn_{options.dataset}_{options.run_name}_{year_start}-{year_end}.nc'
    )

    if output_path.exists():
        if DO_ASSESS:
            assess(output_path, get_damages_xr(options.dataset, year_start, year_end),
                   ignore_removed=False, relax_days=False, prob_threshold=0.5)
        return

    damages, _ = get_damages(options.dataset, year_start, year_end)
    contracts_number = get_contracts_number(damages)
    domain, xs, ys, ds_pred = create_prediction_dataset(year_start, year_end)

    cnn = ImpactCnn(options)
    cnn.set_model(cnn_model)
    cnn.set_precipitation(cpc)
    features = None
    if cnn.options.use_static_attributes or cnn.options.use_event_attributes:
        cnn.select_features(cnn.options.replace_simple_features)
        features = cnn.get_all_features(cnn.options.simple_feature_classes)

    precip_stats = xr.open_dataset(PRECIP_STATS_PATH)
    dg = cnn.get_data_generator_inference(
        events=events,
        features=features,
        exposure=contracts_number,
        precip_stats=precip_stats,
    )
    precip_stats.close()

    for i_x, x in enumerate(tqdm(xs, desc="Progress:", position=0)):
        for i_y, y in enumerate(ys):
            cell_id = domain.cids['ids_map'][i_y, i_x]
            if cell_id == 0:
                ds_pred['predict'][:, i_y, i_x] = np.nan
                continue

            cell_events = events[events['cid'] == cell_id]
            if len(cell_events) == 0:
                continue

            exposure_cid = contracts_number[contracts_number['cid'] == cell_id]
            if len(exposure_cid) == 0 or exposure_cid['nb_contracts'].values[0] == 0:
                ds_pred['predict'][:, i_y, i_x] = np.nan
                continue

            x_input, _ = dg.get_batch_for_cid(cell_id)
            y_pred = cnn.model.predict(x_input, verbose=0).squeeze()
            assert len(y_pred) == len(cell_events)

            for i, (_, event) in enumerate(cell_events.iterrows()):
                if y_pred[i] == 0:
                    continue
                ref_date = pd.to_datetime(event['i_max_date']).replace(hour=0, minute=0)
                ds_pred['predict'].loc[dict(time=ref_date, y=y, x=x)] = y_pred[i]

    ds_pred.to_netcdf(output_path)
    logger.info("Results saved to %s", output_path)
    ds_pred.close()

    if DO_ASSESS:
        assess(output_path, get_damages_xr(options.dataset, year_start, year_end),
               ignore_removed=False, relax_days=False, prob_threshold=0.5)


if __name__ == '__main__':
    main()
