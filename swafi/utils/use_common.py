import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from swafi.config import Config
from swafi.domain import Domain
from swafi.damages_mobiliar import DamagesMobiliar
from swafi.damages_gvz import DamagesGvz
from swafi.utils.event_extraction import extract_events_parallel
from swafi.utils.verification import (
    compute_confusion_matrix,
    print_classic_scores,
    prepare_full_domain_assessment,
)

logger = logging.getLogger(__name__)

config = Config()


def get_damages(dataset, year_start, year_end):
    """Load damages and apply category selection. Returns (damages, removed_claims)."""
    if dataset == 'mobiliar':
        exposure_categories = ['external']
        claim_categories = ['external', 'pluvial']
        damages = DamagesMobiliar(
            dir_exposure=config.get('DIR_EXPOSURE_MOBILIAR'),
            dir_claims=config.get('DIR_CLAIMS_MOBILIAR'),
            year_start=year_start,
            year_end=year_end,
        )
    elif dataset == 'gvz':
        exposure_categories = ['all_buildings']
        claim_categories = ['likely_pluvial']
        damages = DamagesGvz(
            dir_exposure=config.get('DIR_EXPOSURE_GVZ'),
            dir_claims=config.get('DIR_CLAIMS_GVZ'),
            year_start=year_start,
            year_end=year_end,
        )
    else:
        raise ValueError(f"Unknown damage dataset: {dataset}")
    removed_claims = damages.select_categories_type(exposure_categories, claim_categories)
    return damages, removed_claims


def get_damages_xr(dataset, year_start, year_end):
    """Load damages as an xarray Dataset."""
    damages, removed_claims = get_damages(dataset, year_start, year_end)
    return damages.to_xarray(removed_claims=removed_claims)


def assess(result_path, ds_damages, ignore_removed=True, relax_days=True,
           prob_threshold=None):
    """Compute and print classic verification scores.

    prob_threshold=None treats predictions as binary (> 0); otherwise applies
    a probability threshold (>= threshold).
    """
    ds_pred = xr.open_dataset(result_path)
    y_true, y_pred = prepare_full_domain_assessment(
        ds_pred, ds_damages, ignore_removed, relax_days, flatten=True
    )
    if prob_threshold is None:
        y_pred = (y_pred > 0).astype(int)
    else:
        y_pred = (y_pred >= prob_threshold).astype(int)
    y_true = (y_true > 0).astype(int)
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
    print_classic_scores(tp, tn, fp, fn)
    logger.info("*************************************")
    ds_pred.close()


def get_contracts_number(damages):
    """Aggregate mean contract count per cell ID from exposure data."""
    contracts_number = damages.exposure[['cid', 'year', 'selection']].copy()
    contracts_number.rename(columns={'selection': 'nb_contracts'}, inplace=True)
    contracts_number = (
        contracts_number.groupby('cid')
        .agg({'nb_contracts': 'mean'})
        .reset_index()
    )
    return contracts_number


def create_prediction_dataset(year_start, year_end, fill_value=0.0):
    """Create an empty xarray prediction dataset spanning the given year range.

    Returns (domain, xs, ys, ds_pred).
    """
    domain = Domain()
    xs = domain.get_x_axis()
    ys = domain.get_y_axis()
    time = pd.date_range(f'{year_start}-01-01', f'{year_end}-12-31', freq='D')
    results = np.full((len(time), len(ys), len(xs)), fill_value, dtype=np.float32)
    ds_pred = xr.Dataset(
        {'predict': (('time', 'y', 'x'), results)},
        coords={'time': time, 'x': xs, 'y': ys},
    )
    return domain, xs, ys, ds_pred


def get_events(year_start, year_end, event_method, simple_strict_mode=False,
               filter_size=None, precip_dataset='hourly'):
    """Return events DataFrame, loading from pickle cache or extracting in parallel."""
    # The simple method exists for both precipitation datasets: name the cache
    # explicitly; the classic method relies on hourly data (untagged).
    precip_suffix = f'_{precip_dataset}' if event_method == 'simple' else ''
    events_path = (
        Path(config.get('TMP_DIR'))
        / f'test_events_{event_method}{precip_suffix}_{year_start}-{year_end}.pickle'
    )
    if not events_path.exists():
        logger.info("Extracting events and saving to %s...", events_path)
        events = extract_events_parallel(year_start, year_end, event_method,
                                         simple_strict_mode, filter_size,
                                         precip_dataset=precip_dataset)
        events.to_pickle(events_path)
    else:
        events = pd.read_pickle(events_path)
    return events
