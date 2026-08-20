"""
Tests that the precipitation transforms are applied only once. The train,
validation and test data generators share a single precipitation object and
each request the transforms in their constructor; re-applying them would both
corrupt the data and drop the preloaded per-CID time series.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from swafi.precip_archive import PrecipitationArchive


@pytest.fixture
def archive():
    times = pd.date_range('2020-01-01', periods=24, freq='h')
    ys, xs = np.arange(2.0), np.arange(3.0)
    rng = np.random.RandomState(0)
    values = np.abs(rng.normal(size=(len(times), len(ys), len(xs))))

    p = PrecipitationArchive.__new__(PrecipitationArchive)
    p.data = xr.Dataset(
        {'precip': (('time', 'y', 'x'), values)},
        coords={'time': times, 'y': ys, 'x': xs})
    p.precip_var = 'precip'
    p.time_axis_dim, p.y_axis_dim, p.x_axis_dim = 'time', 'y', 'x'
    p._transform_tag = ''
    p.cid_time_series = None
    p.full_grid_data = None

    return p, values


def test_log_transform_applied_once(archive):
    p, values = archive
    p.log_transform()
    p.log_transform()
    np.testing.assert_allclose(
        p.data['precip'].to_numpy(), np.log1p(values).astype('float32'),
        rtol=1e-6)
    assert p._transform_tag == '_log'


def test_normalize_applied_once(archive):
    p, values = archive
    q99 = np.full((2, 3), 2.0)
    p.normalize(q99)
    p.normalize(np.full((2, 3), 5.0))  # must be ignored
    np.testing.assert_allclose(
        p.data['precip'].to_numpy(), (values / 2.0).astype('float32'),
        rtol=1e-6)


def test_standardize_applied_once(archive):
    p, values = archive
    mean, std = np.full((2, 3), 1.0), np.full((2, 3), 2.0)
    p.standardize(mean, std)
    p.standardize(mean, std)
    np.testing.assert_allclose(
        p.data['precip'].to_numpy(), ((values - 1.0) / 2.0).astype('float32'),
        rtol=1e-6)


def test_repeated_transform_keeps_preloaded_series(archive):
    """The per-CID series are preloaded after the transforms; a second
    (no-op) request must not drop them, or every event read falls back to the
    lazy store."""
    p, _ = archive
    p.log_transform()
    p.normalize(np.full((2, 3), 2.0))

    p.cid_time_series = xr.DataArray([1.0, 2.0], dims='cid')
    p.log_transform()
    p.normalize(np.full((2, 3), 2.0))

    assert p.cid_time_series is not None
