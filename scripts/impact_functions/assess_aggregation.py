"""
Test script for loading and evaluating different pre-trained models.
"""
import logging
import numpy as np
import xarray as xr
import geopandas as gpd
from pathlib import Path

from swafi.config import Config
from swafi.damages_mobiliar import DamagesMobiliar
from swafi.damages_gvz import DamagesGvz
from swafi.utils.verification import compute_confusion_matrix, print_classic_scores, prepare_full_domain_assessment
from swafi.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

AGGREGATION_ZIP = R"C:\Data\Data\GIS\Administration\CH_zip_codes\AMTOVZ_ZIP.shp"
AGGREGATION_CATCH = R"C:\Data\Projects\2024 SWF\Data\GIS\Catchments\ezgg_40km2.shp"

DO_ASSESS = True
DATASET = 'mobiliar'  # 'mobiliar' or 'gvz'
PREDICTION_FILE = R"C:\Data\Projects\2024 SWF\Analyses\08 Independent predictions\pred_rf_2023-2024.nc"
AGGREGATION = AGGREGATION_CATCH
DAMAGES_FILE = R"C:\Data\Projects\2024 SWF\Analyses\08 Independent predictions\damages_mobiliar_2023_2024.nc"

config = Config()


def _build_region_map(ds_like, shapefile_path):
    """
    Build a 2D array mapping each grid cell to a polygon region id.

    Parameters
    ----------
    ds_like: xr.Dataset
        Dataset with coordinates 'x' and 'y' and dims ('y', 'x').
    shapefile_path: str|Path
        Path to polygon shapefile.

    Returns
    -------
    region_map: np.ndarray[int]
        2D array (y, x) with integer region ids; -1 for cells outside polygons.
    region_ids: list[int]
        Sorted list of unique region ids present in the grid (>=0).
    region_labels: dict[int, str]
        Optional labels per region id (uses shapefile index if no obvious name).
    """
    gdf = gpd.read_file(shapefile_path)
    # Ensure we have a numeric region id we control
    gdf = gdf.reset_index().rename(columns={'index': 'region_id'})

    xs = ds_like['x'].values
    ys = ds_like['y'].values
    ny, nx = len(ys), len(xs)

    # Build grid points at cell centers
    XX, YY = np.meshgrid(xs, ys)
    flat_x = XX.ravel()
    flat_y = YY.ravel()

    # Align CRS by assuming grid coordinates are in the same CRS as the shapefile
    points = gpd.GeoDataFrame(
        {
            'flat_index': np.arange(flat_x.size, dtype=np.int64),
            'row': np.repeat(np.arange(ny), nx),
            'col': np.tile(np.arange(nx), ny),
        },
        geometry=gpd.points_from_xy(flat_x, flat_y),
        crs=gdf.crs,
    )

    # Spatial join: assign region_id to each point within a polygon
    joined = gpd.sjoin(points, gdf[['region_id', 'geometry']], how='left', predicate='within')

    region_map = np.full((ny, nx), -1, dtype=np.int32)
    not_null = joined['region_id'].notna()
    region_map[joined.loc[not_null, 'row'].to_numpy(), joined.loc[not_null, 'col'].to_numpy()] = (
        joined.loc[not_null, 'region_id'].astype(np.int32).to_numpy()
    )

    # Build region labels (best-effort)
    label_col = None
    for cand in ['ZIP', 'ZIP4']:
        if cand in gdf.columns:
            label_col = cand
            break
    if label_col is None:
        # Fallback to region_id as string
        region_labels = {int(row.region_id): str(int(row.region_id)) for _, row in gdf[['region_id']].iterrows()}
    else:
        region_labels = {int(row.region_id): str(row[label_col]) for _, row in gdf[['region_id', label_col]].iterrows()}

    # Unique region ids that intersect the grid
    region_ids = sorted([int(v) for v in np.unique(region_map) if v >= 0])

    return region_map, region_ids, region_labels


def _aggregate_by_regions(ds_pred, ds_damages, region_map, region_ids, ignore_removed=True, relax_days=True):
    """
    Aggregate predictions and damages per region per day.

    - Predictions: max over pixels in region (per day)
    - Damages: occurrence (1 if any claim > 0 in region that day, else 0)

    Returns
    -------
    y_true_reg: np.ndarray[int]
        Shape (time, n_regions)
    y_pred_reg: np.ndarray[float]
        Shape (time, n_regions)
    """
    time = ds_pred['time'].values
    assert np.array_equal(time, ds_damages['time'].values), "Prediction and damages time axes differ"

    n_time = len(time)
    n_regions = len(region_ids)

    # Precompute flat indices per region for fast reduction
    flat_region = region_map.ravel()
    region_to_idx = {rid: np.where(flat_region == rid)[0] for rid in region_ids}

    y_true_reg = np.zeros((n_time, n_regions), dtype=np.float32) * np.nan
    y_pred_reg = np.zeros((n_time, n_regions), dtype=np.float32) * np.nan

    has_removed = 'removed_claims' in ds_damages.variables

    for t in range(n_time):
        pred2d = ds_pred['predict'].isel(time=t).to_numpy()
        clm2d = ds_damages['claims'].isel(time=t).to_numpy()

        if has_removed and ignore_removed:
            rem2d = ds_damages['removed_claims'].isel(time=t).to_numpy()
            # Mask out removed claims: set to NaN to ignore
            clm2d = clm2d.astype(float)
            clm2d[rem2d == 1] = np.nan

        pflat = pred2d.ravel()
        cflat = clm2d.ravel()

        # Binary occurrence per cell
        cbin = np.zeros_like(cflat, dtype=np.uint8)
        valid = ~np.isnan(cflat)
        cbin[valid & (cflat > 0)] = 1

        for r_idx, rid in enumerate(region_ids):
            idxs = region_to_idx[rid]
            if idxs.size == 0:
                continue
            # Prediction: max prob in region (ignore NaNs)
            pvals = pflat[idxs]
            if pvals.size == 0:
                pred_val = np.nan
            else:
                # If all NaN, keep NaN
                if np.isnan(pvals).all():
                    pred_val = np.nan
                else:
                    pred_val = np.nanmax(pvals)
            # Truth: any claim in region that day
            tvals = cbin[idxs]
            true_val = 1 if np.any(tvals == 1) else 0

            y_pred_reg[t, r_idx] = pred_val
            y_true_reg[t, r_idx] = true_val

    # Optional relax of predictions by +/-1 day around claim days (per region)
    if relax_days:
        for r in range(n_regions):
            # Work on copies to avoid interfering across shifts
            yp = y_pred_reg[:, r].copy()
            yt = y_true_reg[:, r]
            for day in np.where(yt > 0)[0]:
                if np.isnan(yp[day]) or yp[day] > 0:
                    continue
                start_day = max(0, day - 1)
                end_day = min(n_time, day + 1 + 1)  # exclusive end
                # any positive prediction in window?
                if np.any(yp[start_day:end_day] > 0):
                    # emulate pixel-level logic: move value from end/start if that day is false in truth
                    if yt[end_day - 1] == 0 and yp[end_day - 1] > 0:
                        y_pred_reg[day, r] = yp[end_day - 1]
                        y_pred_reg[end_day - 1, r] = 0
                    elif yt[start_day] == 0 and yp[start_day] > 0:
                        y_pred_reg[day, r] = yp[start_day]
                        y_pred_reg[start_day, r] = 0

    return y_true_reg, y_pred_reg


def assess(result_path, ds_damages, ignore_removed=False, relax_days=False, prob_threshold=0.5):
    ds_pred = xr.open_dataset(result_path)

    if AGGREGATION:
        if not Path(AGGREGATION).exists():
            raise FileNotFoundError(f"Aggregation shapefile {AGGREGATION} not found.")

        # Aggregate per polygon region
        region_map, region_ids, region_labels = _build_region_map(ds_pred, AGGREGATION)

        if len(region_ids) == 0:
            raise ValueError("Aggregation shapefile did not overlap the grid at all.")

        y_true_reg, y_pred_reg = _aggregate_by_regions(
            ds_pred, ds_damages, region_map, region_ids,
            ignore_removed=ignore_removed, relax_days=relax_days
        )

        # Flatten and filter NaNs
        y_pred = y_pred_reg.ravel()
        y_true = y_true_reg.ravel()
        mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        if y_pred.size == 0:
            raise ValueError("After aggregation and filtering, no valid samples remain.")

    else:
        # Pixel-level assessment
        y_true, y_pred = prepare_full_domain_assessment(ds_pred, ds_damages, ignore_removed, relax_days, flatten=True)
        logger.info("Pixel-level assessment: %s samples.", y_pred.size)

    y_pred = (y_pred >= prob_threshold).astype(int)
    y_true = (y_true > 0).astype(int)
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
    print_classic_scores(tp, tn, fp, fn)
    logger.info("*************************************")
    ds_pred.close()


def get_damages(dataset):
    if dataset == 'mobiliar':
        damages = DamagesMobiliar(
            year_start=config.get('YEAR_START_TEST'),
            year_end=config.get('YEAR_END_TEST')
        )
    elif dataset == 'gvz':
        damages = DamagesGvz(
            year_start=config.get('YEAR_START_TEST'),
            year_end=config.get('YEAR_END_TEST')
        )
    else:
        raise ValueError(f"Unknown damage dataset: {dataset}")

    return damages


def get_damages_xr(dataset):
    damages = get_damages(dataset)
    claims_xr = damages.from_nc_file(DAMAGES_FILE)

    return claims_xr


def main():
    setup_logging(script_name='assess_aggregation')
    output_path = Path(PREDICTION_FILE)

    if not output_path.exists():
        raise (f"Prediction file {output_path} does not exist. "
               f"Please run the prediction script first.")

    damages = get_damages_xr(DATASET)
    assess(output_path, damages)


if __name__ == '__main__':
    main()
