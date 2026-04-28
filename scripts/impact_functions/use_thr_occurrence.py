"""
Test script for loading and evaluating different pre-trained models.
"""
import logging
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm

from swafi.config import Config
from swafi.domain import Domain
from swafi.impact_basic_options import ImpactBasicOptions
from swafi.impact_thr import ImpactThresholds
from swafi.precip_combiprecip import CombiPrecip
from swafi.damages_mobiliar import DamagesMobiliar
from swafi.damages_gvz import DamagesGvz
from swafi.utils.verification import compute_confusion_matrix, print_classic_scores, prepare_full_domain_assessment
from swafi.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

DO_ASSESS = True

config = Config()


def assess(result_path, ds_damages, ignore_removed=True, relax_days=True):
    ds_pred = xr.open_dataset(result_path)
    y_true, y_pred = prepare_full_domain_assessment(ds_pred, ds_damages, ignore_removed, relax_days, flatten=True)
    y_true = (y_true > 0).astype(int)
    y_pred = (y_pred > 0).astype(int)
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
    print_classic_scores(tp, tn, fp, fn)
    logger.info("*************************************")
    ds_pred.close()


def get_damages_xr(dataset):
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

    claims_xr = damages.to_xarray(removed_claims=removed_claims)

    return claims_xr


def main():
    setup_logging(script_name='use_thr_occurrence')
    options = ImpactBasicOptions()
    options.parse_args()
    options.print_options()
    assert options.is_ok()
    assert options.event_method in ['simple', 'classic'], "Invalid event method."

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

    for method in ['union', 'intersection']:
        output_path = Path(config.get('OUTPUT_DIR')) / f'pred_thr_2019_{options.dataset}_{method}_{year_start}-{year_end}.nc'

        logger.info("Processing %s...", method)

        if output_path.exists():
            if DO_ASSESS:
                damages = get_damages_xr(options.dataset)
                assess(output_path, damages)
            continue

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

        # Create the impact function
        thr = ImpactThresholds(options)

        # Restrict the features to the ones used in the 2019 method
        thr.tabular_features = {'event': ['i_max_q', 'p_sum_q']}
        thr.set_thresholds(thr_i_max=0.9, thr_p_sum=0.98, method=method)

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

                # Predict for the events
                thr.set_events(cell_events)
                y_pred = thr.predict()
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
        logger.info("Results saved to %s", output_path)
        ds_pred.close()

        if DO_ASSESS:
            damages = get_damages_xr(options.dataset)
            assess(output_path, damages)


if __name__ == '__main__':
    main()
