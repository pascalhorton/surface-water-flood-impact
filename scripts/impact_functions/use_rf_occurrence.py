"""
Test script for loading and evaluating different pre-trained models.
"""
import logging
import pickle
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm

from swafi.config import Config
from swafi.domain import Domain
from swafi.impact_rf_options import ImpactRFOptions
from swafi.impact_rf import ImpactRandomForest
from swafi.precip_combiprecip import CombiPrecip
from swafi.damages_mobiliar import DamagesMobiliar
from swafi.damages_gvz import DamagesGvz
from swafi.utils.verification import compute_confusion_matrix, print_classic_scores, prepare_full_domain_assessment
from swafi.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

DO_ASSESS = True
MODEL = R"C:\Users\phorton\Documents\SWF\outputs\model_rf_2025-08-29_160926.pkl"

config = Config()


def assess(result_path, ds_damages, ignore_removed=True, relax_days=True, prob_threshold=0.65):
    ds_pred = xr.open_dataset(result_path)
    y_true, y_pred = prepare_full_domain_assessment(ds_pred, ds_damages, ignore_removed, relax_days, flatten=True)
    y_pred = (y_pred >= prob_threshold).astype(int)
    y_true = (y_true > 0).astype(int)
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
    print_classic_scores(tp, tn, fp, fn)
    logger.info("*************************************")
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
            dir_exposure=config.get('DIR_EXPOSURE_GVZ'),
            dir_claims=config.get('DIR_CLAIMS_GVZ'),
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
    setup_logging(script_name='use_rf_occurrence')
    options = ImpactRFOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()
    assert options.event_method in ['simple', 'classic'], "Invalid event method."

    # Load the model
    rf_model = pickle.load(open(MODEL, "rb"))

    # Extract precipitation events
    year_start = config.get('YEAR_START_TEST')
    year_end = config.get('YEAR_END_TEST')
    events_filename = f'test_events_{options.event_method}_{year_start}-{year_end}.pickle'
    events_path = Path(config.get('TMP_DIR')) / events_filename

    if not events_path.exists():
        logger.info("Extracting events and saving to %s...", events_path)
        cpc = CombiPrecip(year_start, year_end)
        cpc.open_files(config.get('DIR_PRECIP'))
        logger.info("Applying smoothing...")
        cpc.apply_smoothing(filter_size=3)
        events = cpc.extract_events(method=options.event_method)
        events.to_pickle(events_path)
    else:
        events = pd.read_pickle(events_path)

    output_path = Path(config.get('OUTPUT_DIR')) / f'pred_rf_{options.dataset}_{options.run_name}_{year_start}-{year_end}.nc'

    if output_path.exists():
        if DO_ASSESS:
            damages = get_damages_xr(options.dataset)
            assess(output_path, damages)
        return

    # Create resulting xarray dataset
    domain = Domain()
    xs = domain.get_x_axis()
    ys = domain.get_y_axis()
    time = pd.date_range(f'{year_start}-01-01', f'{year_end}-12-31', freq='D')
    results = np.zeros((len(time), len(ys), len(xs)), dtype=np.float32)
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
    rf = ImpactRandomForest(options)
    rf.set_model(rf_model)
    rf.select_features(rf.options.replace_simple_features)
    features = rf.get_all_features(rf.options.simple_feature_classes)

    # Evaluate on all domain cells
    for i_x, x in enumerate(tqdm(xs, desc="Progress:", position=0)):
        for i_y, y in enumerate(ys):
            cell_id = domain.cids['ids_map'][i_y, i_x]
            if cell_id == 0:
                ds_pred['predict'][:, i_y, i_x] = np.nan
                continue

            # Get events for this cell
            cell_events = events[events['cid'] == cell_id]
            if len(cell_events) == 0:
                continue

            features_cid = features[features['cid'] == cell_id]
            if len(features_cid) == 0:
                ds_pred['predict'][:, i_y, i_x] = np.nan
                continue

            exposure_cid = contracts_number[contracts_number['cid'] == cell_id]
            if len(exposure_cid) == 0 or exposure_cid['nb_contracts'].values[0] == 0:
                ds_pred['predict'][:, i_y, i_x] = np.nan
                continue

            # Predict for the events
            rf.set_events(cell_events)
            rf.set_features(features_cid)
            rf.set_exposure(exposure_cid)

            rf.df.dropna(subset=rf.features, inplace=True)
            x_input = rf.df[rf.features].to_numpy()
            y_pred = rf.model.predict_proba(x_input)[:, 1]  # Probability of class 1
            assert len(y_pred) == len(rf.df)

            # Loop over events and store the target value at the correct date
            for i, (_, row) in enumerate(rf.df.iterrows()):
                if y_pred[i] == 0:
                    continue
                # Store the prediction at the date of the event
                ref_date = pd.to_datetime(row['i_max_date']).replace(hour=0, minute=0)
                ds_pred['predict'].loc[dict(time=ref_date, y=y, x=x)] = y_pred[i]

    # Save the results
    ds_pred.to_netcdf(output_path)
    logger.info("Results saved to %s", output_path)
    ds_pred.close()

    if DO_ASSESS:
        damages = get_damages_xr(options.dataset)
        assess(output_path, damages)


if __name__ == '__main__':
    main()
