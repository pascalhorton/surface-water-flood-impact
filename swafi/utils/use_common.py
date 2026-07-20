import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from swafi.config import Config
from swafi.domain import Domain
from swafi.damages_mobiliar import DamagesMobiliar
from swafi.damages_gvz import DamagesGvz
from swafi.precip_combiprecip import CombiPrecip
from swafi.precip_combiprecip_5min import CombiPrecip5min
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


class GridPredictionWriter:
    """
    Vectorized writes of per-event predictions into a (time, y, x) prediction
    dataset. Replaces the per-cell loops over the domain grid: cell masking and
    event writes are resolved through a cid -> (i_y, i_x) lookup built once.
    """

    def __init__(self, ds_pred, domain):
        self._pred = ds_pred['predict'].values
        ids_map = domain.cids['ids_map']
        self._outside = ids_map == 0
        i_y, i_x = np.nonzero(ids_map)
        self._pos = pd.DataFrame({'i_y': i_y, 'i_x': i_x},
                                 index=ids_map[i_y, i_x])
        self._t0 = ds_pred['time'].values[0]

    def get_map_cids(self):
        """Return the set of cell ids present in the domain map."""
        return set(self._pos.index)

    def mask_outside_domain(self):
        """Set all cells absent from the domain map (cid == 0) to NaN."""
        self._pred[:, self._outside] = np.nan

    def fill_cells(self, cids, value):
        """Set whole cells (all time steps) to a constant (NaN mask or background)."""
        cids = np.fromiter(cids, dtype=np.int64)
        if len(cids) == 0:
            return
        pos = self._pos.loc[cids]
        self._pred[:, pos['i_y'].to_numpy(), pos['i_x'].to_numpy()] = value

    def write_events(self, cids, dates, values):
        """
        Write the nonzero predictions at the day of each event. Row order is
        preserved, so on same-day collisions within a cell the last nonzero
        prediction wins, like the original per-event loops (which skipped
        zeros and overwrote previous writes).
        """
        values = np.asarray(values, dtype=float)
        cids = np.asarray(cids, dtype=np.int64)
        keep = values != 0
        if not keep.any():
            return
        days = pd.DatetimeIndex(np.asarray(dates)[keep]).normalize()
        t_idx = ((days.values - self._t0) // np.timedelta64(1, 'D')).astype(int)
        assert (t_idx >= 0).all() and (t_idx < self._pred.shape[0]).all(), \
            "Event date outside the prediction period."
        pos = self._pos.loc[cids[keep]]
        self._pred[t_idx, pos['i_y'].to_numpy(), pos['i_x'].to_numpy()] = values[keep]


def predict_events_in_chunks(model, dg, idxs, chunk_size=1024):
    """
    Run a keras model over the given event indices of an inference data
    generator, many cells per predict() call. One call per grid cell is
    dominated by the fixed per-call overhead; chunking removes it while
    keeping the memory footprint bounded (relevant for CNN inputs).

    Parameters
    ----------
    model : keras.Model
        The trained model.
    dg : ImpactDlDataGenerator
        The inference data generator holding all events.
    idxs : np.ndarray
        The event indices (into the generator's full data) to predict.
    chunk_size : int
        The number of events per predict() call.

    Returns
    -------
    np.ndarray
        The predictions, aligned with idxs.
    """
    y_pred = np.empty(len(idxs), dtype=np.float32)
    for start in tqdm(range(0, len(idxs), chunk_size), desc="Predicting"):
        sel = idxs[start:start + chunk_size]
        x, _ = dg.get_batch_for_indices(sel)
        y_pred[start:start + len(sel)] = np.asarray(
            model.predict(x, verbose=0)).reshape(-1)
    return y_pred


def ensure_precip_dataset(options, default='hourly'):
    """Backfill the precipitation dataset on options restored from a model.

    Models trained before the precip_dataset option existed carry no such
    attribute: assume the default rather than failing. Returns the value.

    Parameters
    ----------
    options : ImpactBasicOptions
        The options, typically restored from a saved model.
    default : str
        The dataset to assume when the option is absent.
    """
    if getattr(options, 'precip_dataset', None) is None:
        logger.warning(
            "The loaded model carries no precipitation dataset option: assuming "
            "'%s'. Retrain the model or set options.precip_dataset explicitly if "
            "it was trained on another dataset.", default)
        options.precip_dataset = default

    return options.precip_dataset


def create_precipitation(precip_dataset, year_start, year_end):
    """Instantiate the precipitation source for the given dataset name.

    The data itself is read lazily from the corresponding zarr store.

    Parameters
    ----------
    precip_dataset : str
        The precipitation dataset ('hourly' or '5min').
    year_start : int
        The first year to cover.
    year_end : int
        The last year to cover.
    """
    if precip_dataset == 'hourly':
        return CombiPrecip(year_start, year_end)
    elif precip_dataset == '5min':
        return CombiPrecip5min(year_start, year_end)

    raise ValueError(f"Unknown precipitation dataset: {precip_dataset}")


def get_events(year_start, year_end, event_method,
               filter_size=None, precip_dataset='hourly', detection_window_h=1.0):
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
        events = extract_events_parallel(year_start, year_end, event_method, filter_size,
                                         precip_dataset=precip_dataset,
                                         detection_window_h=detection_window_h)
        events.to_pickle(events_path)
    else:
        events = pd.read_pickle(events_path)
    return events
