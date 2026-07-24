from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

zarr = pytest.importorskip('zarr')

from swafi.precip_combiprecip import CombiPrecip

# Small synthetic grid (pixel centres, EPSG:2056-like, 1 km resolution)
X_AXIS = 2600500.0 + np.arange(8) * 1000.0
Y_AXIS = 1200500.0 - np.arange(6) * 1000.0
YEAR = 2021


def _make_cpc(tmp_path):
    """Build a CombiPrecip instance without touching config/domain files."""
    cpc = object.__new__(CombiPrecip)
    cpc.dataset_name = 'CombiPrecip'
    cpc.data = None
    cpc.data_path = None
    cpc.x_axis_dim = 'x'
    cpc.y_axis_dim = 'y'
    cpc.time_axis_dim = 'time'
    cpc.precip_var = 'precip'
    cpc.resolution = None
    cpc.time_step = None
    cpc.year_start = YEAR
    cpc.year_end = YEAR
    cpc.native_time_step = 1
    cpc.mem_nb_pixels = 64
    cpc.cid_time_series = None
    cpc.full_grid_data = None
    cpc._transform_tag = ''
    cpc.tmp_dir = tmp_path
    # Domain covering the whole synthetic grid (used for the store crop)
    cpc.domain = SimpleNamespace(cids={'extent': SimpleNamespace(
        left=X_AXIS[0], right=X_AXIS[-1], bottom=Y_AXIS[-1], top=Y_AXIS[0])})
    return cpc


def _write_source_netcdf(tmp_path):
    """
    Two monthly netCDF files (Jan/Feb 2021) in the raw CombiPrecip layout
    (variable 'CPC', time dim 'REFERENCE_TS'), with the dirt the cleaning must
    handle: a duplicated timestamp, a missing timestamp and a leading NaN.
    """
    rng = np.random.default_rng(11)

    def month_ds(times, values):
        return xr.Dataset(
            {'CPC': (('REFERENCE_TS', 'y', 'x'), values.astype('float32'))},
            coords={'REFERENCE_TS': times, 'y': Y_AXIS, 'x': X_AXIS})

    # January: leading NaN step, duplicate at 05:00 on Jan 10, gap at 12:00 on
    # Jan 20 (linear interpolation should fill it with the neighbour mean).
    times_jan = pd.date_range('2021-01-01 00:00', '2021-01-31 23:00', freq='h')
    values_jan = rng.uniform(0.0, 5.0, (len(times_jan), 6, 8))
    values_jan[0] = np.nan  # leading NaN -> 0 (cannot be interpolated)

    gap_idx = times_jan.get_loc(pd.Timestamp('2021-01-20 12:00'))
    expected_gap = (values_jan[gap_idx - 1] + values_jan[gap_idx + 1]) / 2
    keep = np.ones(len(times_jan), dtype=bool)
    keep[gap_idx] = False

    dup_idx = times_jan.get_loc(pd.Timestamp('2021-01-10 05:00'))
    times_jan_dirty = times_jan[keep].insert(dup_idx + 1, times_jan[dup_idx])
    values_jan_dirty = np.insert(values_jan[keep], dup_idx + 1,
                                 999.0, axis=0)  # duplicate with junk values

    # February: clean month
    times_feb = pd.date_range('2021-02-01 00:00', '2021-02-28 23:00', freq='h')
    values_feb = rng.uniform(0.0, 5.0, (len(times_feb), 6, 8))

    month_ds(times_jan_dirty, values_jan_dirty).to_netcdf(
        tmp_path / 'CPC_00060_H_20210101000000_20210131230000.nc')
    month_ds(times_feb, values_feb).to_netcdf(
        tmp_path / 'CPC_00060_H_20210201000000_20210228230000.nc')

    expected_jan = np.nan_to_num(values_jan)  # leading NaN -> 0
    expected_jan[gap_idx] = expected_gap
    return expected_jan, values_feb


@pytest.fixture()
def built_store(tmp_path):
    expected_jan, values_feb = _write_source_netcdf(tmp_path)
    cpc = _make_cpc(tmp_path)
    store = tmp_path / 'cpch_hourly.zarr'
    cpc.build_zarr_store(store, data_path=str(tmp_path))
    return cpc, store, expected_jan, values_feb


def test_build_store_cleans_and_matches_source(built_store):
    _, store, expected_jan, values_feb = built_store

    data = xr.open_zarr(store, consolidated=False)['precip']

    jan = data.sel(time=slice('2021-01-01', '2021-01-31')).values
    np.testing.assert_allclose(jan, expected_jan.astype('float32'), rtol=1e-6)

    feb = data.sel(time=slice('2021-02-01', '2021-02-28')).values
    np.testing.assert_allclose(feb, values_feb.astype('float32'), rtol=1e-6)

    # Months without source data are written as 0 (pickle-compatible)
    assert float(data.sel(time=slice('2021-03-01', None)).max()) == 0.0

    # Full-year calendar and cropped axes
    assert data.sizes['time'] == 365 * 24
    np.testing.assert_allclose(data['x'].values, X_AXIS)
    np.testing.assert_allclose(data['y'].values, Y_AXIS)


def test_completed_build_removes_markers_and_rerun_is_noop(built_store, tmp_path):
    cpc, store, _, _ = built_store
    done_dir = tmp_path / 'cpch_hourly.zarr.done'
    assert not done_dir.exists()  # removed once the build completed

    files = [p for p in store.rglob('*') if p.is_file()]
    mtimes = {p: p.stat().st_mtime_ns for p in files}
    cpc.build_zarr_store(store, data_path=str(tmp_path))  # no-op rerun
    assert not done_dir.exists()
    assert all(p.stat().st_mtime_ns == mtimes[p] for p in files)  # nothing rewritten


def test_build_resumes_only_missing_months(built_store, tmp_path):
    cpc, store, expected_jan, _ = built_store

    # Simulate an interrupted build: every month but January marked as done
    done_dir = tmp_path / 'cpch_hourly.zarr.done'
    done_dir.mkdir()
    months = pd.date_range('2021-01-01', '2021-12-31', freq='MS')
    for month in months[1:]:
        (done_dir / month.strftime('%Y-%m')).touch()

    cpc.build_zarr_store(store, data_path=str(tmp_path))
    assert not done_dir.exists()

    data = xr.open_zarr(store, consolidated=False)['precip']
    jan = data.sel(time=slice('2021-01-01', '2021-01-31')).values
    np.testing.assert_allclose(jan, expected_jan.astype('float32'), rtol=1e-6)


def test_derived_daily_store(built_store):
    cpc, store, _, _ = built_store

    cpc.open_zarr(store)
    base = cpc.data['precip'].compute()
    cpc._use_derived_store(1, 24)

    assert cpc.time_step == 24
    expected = base.resample(time='24h', closed='right', label='right').sum('time')
    np.testing.assert_allclose(cpc.data['precip'].values, expected.values,
                               rtol=1e-5)

    # The derived store is materialized once and reused (name carries the period)
    derived_path = cpc.tmp_dir / 'precip_combiprecip_r1_t1440min_2021-2021.zarr'
    assert (derived_path / 'zarr.json').exists()


def test_stats_and_lazy_transforms(built_store):
    cpc, store, _, _ = built_store

    cpc.open_zarr(store)
    cpc.mem_nb_pixels = 3  # force several spatial blocks
    raw = cpc.data['precip'].compute().values

    mean, std = cpc.compute_mean_and_std_per_pixel()
    q99 = cpc.compute_quantile_per_pixel(0.99)
    np.testing.assert_allclose(mean, np.nanmean(raw, axis=0), rtol=1e-5)
    np.testing.assert_allclose(std, np.nanstd(raw, axis=0), rtol=1e-5)
    np.testing.assert_allclose(q99, np.nanquantile(raw, 0.99, axis=0), rtol=1e-5)

    # Stats are cached on disk: a second call reloads identical values
    q99_again = cpc.compute_quantile_per_pixel(0.99)
    np.testing.assert_allclose(q99_again, q99)

    # Lazy log transform, then normalization by the (log) q99
    cpc.log_transform()
    q99_log = cpc.compute_quantile_per_pixel(0.99)
    np.testing.assert_allclose(q99_log, np.nanquantile(np.log1p(raw), 0.99, axis=0),
                               rtol=1e-5)
    cpc.normalize(q99_log)

    chunk = cpc.get_data_chunk(
        pd.Timestamp('2021-01-05'), pd.Timestamp('2021-01-07'),
        X_AXIS[2], X_AXIS[4], Y_AXIS[1], Y_AXIS[3])
    raw_da = xr.DataArray(
        raw, coords={'time': pd.date_range('2021-01-01', periods=raw.shape[0],
                                           freq='h'),
                     'y': Y_AXIS, 'x': X_AXIS}, dims=('time', 'y', 'x'))
    expected = np.log1p(raw_da.sel(
        time=slice(pd.Timestamp('2021-01-05'), pd.Timestamp('2021-01-07')),
        x=slice(X_AXIS[2], X_AXIS[4]),
        y=slice(Y_AXIS[1], Y_AXIS[3]))).values / q99_log[1:4, 2:5]
    np.testing.assert_allclose(chunk, expected, rtol=1e-5)


def test_select_subdomain_fills_outside_with_nan(built_store):
    cpc, store, _, _ = built_store

    cpc.open_zarr(store)
    x_ext = np.concatenate([X_AXIS, [X_AXIS[-1] + 1000.0]])
    cpc.select_subdomain(x_ext, Y_AXIS)
    field = cpc.data['precip'].isel(time=40).values
    assert field.shape == (6, 9)
    assert np.isnan(field[:, -1]).all()
    assert not np.isnan(field[:, :-1]).any()


def test_preload_all_cid_data(built_store):
    cpc, store, _, _ = built_store

    cpc.open_zarr(store)
    coords = {101: (float(X_AXIS[2]), float(Y_AXIS[1])),
              102: (float(X_AXIS[5]), float(Y_AXIS[4]))}
    cpc.domain = SimpleNamespace(
        get_cid_coordinates=lambda cid: coords[cid],
        cids=cpc.domain.cids)

    cpc.preload_all_cid_data([101, 102])

    direct = cpc.data['precip'].sel(x=X_AXIS[5], y=Y_AXIS[4]).compute().values
    np.testing.assert_allclose(
        cpc.cid_time_series.sel(cid=102).values, direct, rtol=1e-6)

    # get_data_chunk uses the preloaded series when given a CID
    chunk = cpc.get_data_chunk(
        pd.Timestamp('2021-02-01'), pd.Timestamp('2021-02-02'),
        None, None, None, None, cid=101)
    assert chunk.shape[1:] == (1, 1)

    # The cache file makes a second preload load from disk
    cache_files = list(cpc.tmp_dir.glob('precip_combiprecip_all_cids_*.nc'))
    assert len(cache_files) == 1
    cpc.preload_all_cid_data([101, 102])
    np.testing.assert_allclose(
        cpc.cid_time_series.sel(cid=102).values, direct, rtol=1e-6)
