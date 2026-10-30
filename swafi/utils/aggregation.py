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
    assert np.array_equal(time, ds_damages['time'].values),         "Prediction and damages time axes differ"

    n_time = len(time)
    n_regions = len(region_ids)

    # Map each cell to a region POSITION (0..n_regions-1), -1 outside, then
    # sort the inside cells by it once. A per-region numpy call inside the day
    # loop is 731 x 3952 = 2.9 M small operations for the ZIP aggregation,
    # which does not finish in reasonable time; grouping the cells once and
    # reducing with `reduceat` makes it two calls per day instead.
    flat_region = region_map.ravel()
    lookup = {rid: i for i, rid in enumerate(region_ids)}
    region_pos = np.full(flat_region.shape, -1, dtype=np.int64)
    for rid, i in lookup.items():
        region_pos[flat_region == rid] = i

    inside = np.where(region_pos >= 0)[0]
    order = inside[np.argsort(region_pos[inside], kind='stable')]
    sorted_pos = region_pos[order]
    # Every region in region_ids has at least one cell by construction (they
    # were derived from the grid), so searchsorted gives one start per region
    # and no group is empty - which reduceat could not express if it were.
    starts = np.searchsorted(sorted_pos, np.arange(n_regions), side='left')
    assert len(starts) == n_regions

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

        # Prediction: the max probability in the region. The max is the right
        # reducer for an occurrence question - the region is warned if any cell
        # in it is warned - and a mean would penalise a correct, localised
        # prediction for the size of the polygon it happens to sit in.
        # np.fmax ignores NaN and yields NaN only where every cell is NaN,
        # which is the "region the model could not speak to" case.
        y_pred_reg[t] = np.fmax.reduceat(pflat[order], starts)

        # Truth: 1 if any cell in the region carries a claim that day.
        #
        # A region-day where EVERY cell has NaN claims is NaN, not 0. NaN in
        # `claims` means the cell carries no insurance, so no claim is
        # possible there and the cell-day is not a legitimate negative - which
        # is exactly why the pixel path drops it with
        # `~np.isnan(y_true)`. Scoring such a region as a correct negative
        # fabricates samples, and at the scales used here it fabricates almost
        # all of them: the ZIP shapefile covers Switzerland while GVZ covers
        # one canton, so 3,618 of the 3,952 ZIP regions overlapping the grid
        # (92%) hold no insured cell at all. Counting them put 2.8 M samples
        # in a comparison whose pixel-based twin has 1.29 M, and dropped the
        # base rate from 0.0255% to 0.0094% - which moves every rate-based
        # metric in the table.
        cvals = cflat[order]
        valid = (~np.isnan(cvals)).astype(np.float32)
        has_claim = np.maximum.reduceat(
            (valid > 0) & (cvals > 0), starts).astype(np.float32)
        n_insured = np.add.reduceat(valid, starts)
        y_true_reg[t] = np.where(n_insured > 0, has_claim, np.nan)

    if relax_days:
        n_moved = 0
        for r in range(n_regions):
            n_moved += relax_prediction_days(y_true_reg[:, r], y_pred_reg[:, r])
        logger.info("Relaxation moved %s predictions over %s regions.",
                    n_moved, n_regions)

    return y_true_reg, y_pred_reg
