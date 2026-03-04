"""
Test script for loading and evaluating different pre-trained models.
"""
import keras
import random
import tensorflow as tf
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm

from swafi.config import Config
from swafi.domain import Domain
from swafi.impact_cnn import ImpactCnn
from swafi.damages_mobiliar import DamagesMobiliar
from swafi.damages_gvz import DamagesGvz
from swafi.precip_combiprecip import CombiPrecip
from swafi.utils.verification import compute_confusion_matrix, print_classic_scores, prepare_full_domain_assessment

DO_ASSESS = True
MODEL = R"C:\Users\phorton\Documents\SWF\outputs\model_ann_gvz_70.keras"
DATASET = 'gvz'  # 'mobiliar' or 'gvz'
THRESHOLD = 0.5

config = Config()


def assess(result_path, ds_damages, ignore_removed=True, relax_days=True, prob_threshold=0.5):
    ds_pred = xr.open_dataset(result_path)

    y_true, y_pred = prepare_full_domain_assessment(ds_pred, ds_damages, ignore_removed, relax_days, flatten=True)
    y_pred = (y_pred >= prob_threshold).astype(int)
    y_true = (y_true > 0).astype(int)
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
    print_classic_scores(tp, tn, fp, fn)
    print("*************************************")
    ds_pred.close()


def get_damages(dataset):
    if dataset == 'mobiliar':
        exposure_categories = ['external']
        claim_categories = ['external', 'pluvial']
        damages = DamagesMobiliar(
            dir_exposure=config.get('DIR_EXPOSURE_MOBILIAR'),
            dir_claims=config.get('DIR_CLAIMS_MOBILIAR'),
            year_start=config.get('YEAR_START_TEST'),
            year_end=config.get('YEAR_END_TEST')
        )
    elif dataset == 'gvz':
        exposure_categories = ['all_buildings']
        claim_categories = ['likely_pluvial']
        damages = DamagesGvz(
            dir_exposure=config.get('DIR_EXPOSURE_GVZ') + '/test period/',
            dir_claims=config.get('DIR_CLAIMS_GVZ') + '/test period/',
            year_start=config.get('YEAR_START_TEST'),
            year_end=config.get('YEAR_END_TEST')
        )
    else:
        raise ValueError(f"Unknown damage dataset: {dataset}")

    removed_claims = damages.select_categories_type(exposure_categories, claim_categories)

    return damages, removed_claims


def get_damages_xr(dataset):
    damages, removed_claims = get_damages(dataset)
    claims_xr = damages.to_xarray(removed_claims=removed_claims)

    return claims_xr


def main():
    # Load the keras model
    ann_model = keras.models.load_model(MODEL)

    options = ann_model.options
    options.dataset = DATASET
    options.print_options()
    assert options.is_ok()

    # Clear session and set the seed
    keras.backend.clear_session()
    if options.random_state is not None:
        random.seed(options.random_state)
        np.random.seed(options.random_state)
        tf.random.set_seed(options.random_state)
        keras.utils.set_random_seed(options.random_state)

    if ann_model.model == None:
        ann_model.build_model()

    # Extract precipitation events
    year_start = config.get('YEAR_START_TEST')
    year_end = config.get('YEAR_END_TEST')
    events_filename = f'test_events_{year_start}-{year_end}.pickle'
    events_path = Path(config.get('TMP_DIR')) / events_filename

    if not events_path.exists():
        print(f"Extracting events and saving to {events_path}...")
        cpc = CombiPrecip(year_start, year_end)
        cpc.open_files(config.get('DIR_PRECIP'))
        print("Applying smoothing...")
        cpc.apply_smoothing(filter_size=3)
        events = cpc.extract_events()
        events.to_pickle(events_path)
    else:
        events = pd.read_pickle(events_path)

    output_path = Path(config.get('OUTPUT_DIR')) / f'pred_ann_{options.run_name}_{year_start}-{year_end}.nc'

    if output_path.exists():
        if DO_ASSESS:
            damages = get_damages_xr(options.dataset)
            assess(output_path, damages, prob_threshold=THRESHOLD)
        return

    # Create resulting xarray dataset
    domain = Domain()
    xs = domain.get_x_axis()
    ys = domain.get_y_axis()
    time = pd.date_range(f'{year_start}-01-01', f'{year_end}-12-31', freq='D')
    results = np.full((len(time), len(ys), len(xs)), np.nan, dtype=np.float32)
    ds_pred = xr.Dataset(
        {'predict': (('time', 'y', 'x'), results)},
        coords={'time': time, 'x': xs, 'y': ys}
    )

    # Get the number of contracts
    damages, removed_claims = get_damages(options.dataset)
    contracts_number = damages.exposure[['cid', 'year', 'selection']].copy()
    contracts_number.rename(columns={'selection': 'nb_contracts'}, inplace=True)

    # Average number of contracts over the years (per cid)
    contracts_number = contracts_number.groupby('cid').agg({'nb_contracts': 'mean'}).reset_index()

    # Create the impact function
    ann = ImpactCnn(options)
    ann.set_model(ann_model)
    features = None
    if ann.options.use_static_attributes or ann.options.use_event_attributes:
        ann.select_features(ann.options.replace_simple_features)
        features = ann.get_all_features(ann.options.simple_feature_classes)

    # Prepare the data (normalization)
    dg = ann.get_data_generator_inference(
        events=events,
        features=features,
        exposure=contracts_number
    )

    # Evaluate on all domain cells
    for i_x, x in enumerate(tqdm(xs, desc="Progress:", position=0)):
        for i_y, y in enumerate(ys):
            cell_id = domain.cids['ids_map'][i_y, i_x]
            if cell_id == 0:
                ds_pred['predict'][:, i_y, i_x] = np.nan
                continue

            # Get exposure for this cell
            exposure_cid = contracts_number[contracts_number['cid'] == cell_id]
            if len(exposure_cid) == 0 or exposure_cid['nb_contracts'].values[0] == 0:
                continue

            # Get events for this cell
            cell_events = events[events['cid'] == cell_id]
            if len(cell_events) == 0:
                continue

            # Fill with zeros
            ds_pred['predict'][:, i_y, i_x] = 0

            # Predict
            x_input, _ = dg.get_batch_for_cid(cell_id)
            y_pred = ann.model.predict(x_input, verbose=0)

            # Get rid of the single dimension
            y_pred = y_pred.squeeze()

            assert len(y_pred) == len(cell_events)

            # Loop over events and store the target value at the correct date
            for i, (_, event) in enumerate(cell_events.iterrows()):
                if y_pred[i] == 0:
                    continue
                # Store the prediction at the date of the event
                ref_date = pd.to_datetime(event['i_max_date']).replace(hour=0, minute=0)
                ds_pred['predict'].loc[dict(time=ref_date, y=y, x=x)] = y_pred[i]

    # Save the results
    ds_pred.to_netcdf(output_path)
    print(f"Results saved to {output_path}")
    ds_pred.close()

    if DO_ASSESS:
        damages = get_damages_xr(options.dataset)
        assess(output_path, damages, prob_threshold=THRESHOLD)


if __name__ == '__main__':
    main()
