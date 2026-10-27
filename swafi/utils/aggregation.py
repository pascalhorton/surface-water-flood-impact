"""
Spatial aggregation of gridded predictions and damages.

The full-domain assessment is pixel-based: one sample per cell and day. Many
questions are better asked at a coarser unit - a ZIP region or a catchment is
closer to what an insurer or a warning service acts on, and a prediction that
lands one cell away from the claim is a near miss at 1 km and a hit at
catchment scale.

This module was lifted out of `scripts/impact_functions/assess_aggregation.py`,
where it lived as two private functions. It is library code: two scripts need
it, and while it was a script-local copy its day-relaxation drifted out of
step with the pixel-level one in `swafi.utils.verification` - the two
preferred opposite neighbours for over a week. Both now call
`relax_prediction_days`.
"""
import logging

import geopandas as gpd
import numpy as np

from swafi.utils.verification import relax_prediction_days

logger = logging.getLogger(__name__)


def build_region_map(ds_like, shapefile_path):
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
    joined = gpd.sjoin(points, gdf[['region_id', 'geometry']], how='left',
                       predicate='within')

    region_map = np.full((ny, nx), -1, dtype=np.int32)
    not_null = joined['region_id'].notna()
    region_map[joined.loc[not_null, 'row'].to_numpy(),
               joined.loc[not_null, 'col'].to_numpy()] = (
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
        region_labels = {int(row.region_id): str(int(row.region_id))
                         for _, row in gdf[['region_id']].iterrows()}
    else:
        region_labels = {int(row.region_id): str(row[label_col])
                         for _, row in gdf[['region_id', label_col]].iterrows()}

    # Unique region ids that intersect the grid
    region_ids = sorted([int(v) for v in np.unique(region_map) if v >= 0])

    return region_map, region_ids, region_labels


def aggregate_by_regions(ds_pred, ds_damages, region_map, region_ids,
                         ignore_removed=True, relax_days=True):
    """
    Aggregate predictions and damages per region per day.

    - Predictions: max over the pixels in the region, per day. The max is the
      right reducer for an occurrence question: the region is warned if any
      cell in it is warned, so a mean would penalise a correct, localised
      prediction for the size of the polygon it sits in.
    - Damages: occurrence, 1 if any claim > 0 in the region that day.

    Note this reads `ds_pred` and `ds_damages` day by day rather than pulling
    whole arrays, so unlike the pixel-level path it does NOT modify them in
    place. The relaxation acts on the aggregated series.

    Parameters
    ----------
    ds_pred: xr.Dataset
        The dataset with the predictions.
    ds_damages: xr.Dataset
        The dataset with the damages.
    region_map: np.ndarray[int]
        2D array (y, x) of region ids, as built by `build_region_map`.
    region_ids: list[int]
        The region ids to aggregate over.
    ignore_removed: bool
        Whether to ignore the claims removed by the category selection.
    relax_days: bool
        Whether to allow a prediction one day either side of the claim.

    Returns
    -------
    y_true_reg: np.ndarray
        Shape (time, n_regions).
    y_pred_reg: np.ndarray
        Shape (time, n_regions).
    """
    time = ds_pred['time'].values
    assert np.array_equal(time, ds_damages['time'].values), \
        "Prediction and damages time axes differ"

    n_time = len(time)
    n_regions = len(region_ids)

    # Precompute flat indices per region for fast reduction
    flat_region = region_map.ravel()
    region_to_idx = {rid: np.where(flat_region == rid)[0] for rid in region_ids}

    y_true_reg = np.full((n_time, n_regions), np.nan, dtype=np.float32)
    y_pred_reg = np.full((n_time, n_regions), np.nan, dtype=np.float32)

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
            if pvals.size == 0 or np.isnan(pvals).all():
                pred_val = np.nan
            else:
                pred_val = np.nanmax(pvals)
            # Truth: any claim in region that day
            y_pred_reg[t, r_idx] = pred_val
            y_true_reg[t, r_idx] = 1 if np.any(cbin[idxs] == 1) else 0

    if relax_days:
        n_moved = 0
        for r in range(n_regions):
            n_moved += relax_prediction_days(y_true_reg[:, r], y_pred_reg[:, r])
        logger.info("Relaxation moved %s predictions over %s regions.",
                    n_moved, n_regions)

    return y_true_reg, y_pred_reg
