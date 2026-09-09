"""
Tests for the sub-hourly precipitation time step: the conversion of a time step
given in hours to a whole number of minutes, and the resampling / time-grid
maths that depend on it.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from swafi.impact_cnn_data_generator import ImpactCnnDataGenerator
from swafi.precip_archive import PrecipitationArchive, time_step_to_minutes


@pytest.mark.parametrize("hours, minutes", [
    (1, 60), (2, 120), (0.5, 30), (0.25, 15), (1 / 6, 10),
    (5 / 60, 5), (0.0833, 5), (0.1667, 10),  # rounded decimals of an hour
])
def test_time_step_to_minutes(hours, minutes):
    assert time_step_to_minutes(hours) == minutes


@pytest.mark.parametrize("bad", [0.04, 0.007, 0.02, 0.0])
def test_time_step_to_minutes_rejects_non_integer_minutes(bad):
    with pytest.raises(AssertionError):
        time_step_to_minutes(bad)


def test_resample_5min_to_30min_sums_right_labelled():
    # 4 hours of 5-min steps of 1 mm, timestamps at the END of each interval.
    times = pd.date_range('2020-01-01 00:05', periods=48, freq='5min')
    vals = np.ones((48, 1, 1), dtype='float32')
    p = PrecipitationArchive.__new__(PrecipitationArchive)
    p.data = xr.Dataset({'precip': (('time', 'y', 'x'), vals)},
                        coords={'time': times, 'y': [0.0], 'x': [0.0]})
    p.precip_var = 'precip'
    p.time_axis_dim, p.y_axis_dim, p.x_axis_dim = 'time', 'y', 'x'
    p.resolution = 1
    p.native_time_step = 5 / 60
    p.time_step = 0.5

    out = p._resample(p.data)
    # Each 30-min bin sums six 5-min steps -> 6; labelled at the interval end.
    np.testing.assert_allclose(out['precip'].to_numpy().ravel(), 6.0)
    assert str(out['time'].to_numpy()[0]).startswith('2020-01-01T00:30')


def test_resample_native_step_is_noop():
    times = pd.date_range('2020-01-01 00:05', periods=12, freq='5min')
    vals = np.arange(12, dtype='float32').reshape(12, 1, 1)
    p = PrecipitationArchive.__new__(PrecipitationArchive)
    p.data = xr.Dataset({'precip': (('time', 'y', 'x'), vals)},
                        coords={'time': times, 'y': [0.0], 'x': [0.0]})
    p.precip_var = 'precip'
    p.time_axis_dim, p.y_axis_dim, p.x_axis_dim = 'time', 'y', 'x'
    p.resolution = 1
    p.native_time_step = 5 / 60
    p.time_step = 5 / 60  # equal to native: no aggregation

    out = p._resample(p.data)
    np.testing.assert_array_equal(out['precip'].to_numpy().ravel(), np.arange(12))


@pytest.mark.parametrize("days_before, days_after, step_min, expected", [
    (2, 1, 60, 4 * 24 + 1),       # hourly, the historical default
    (0, 0, 5, 288 + 1),           # one day at 5-min resolution
    (2, 1, 30, 4 * 48 + 1),       # 30-min steps over four days
])
def test_time_dim_size(days_before, days_after, step_min, expected):
    # The generator takes the time step in minutes.
    g = ImpactCnnDataGenerator.__new__(ImpactCnnDataGenerator)
    g.X_precip = object()
    g.time_dim_size = None
    g.precip_days_before = days_before
    g.precip_days_after = days_after
    g.precip_time_step = step_min
    assert g.get_time_dim_size() == expected


# --- the full-grid preload --------------------------------------------------


def test_preload_full_grid_computes_once():
    """The generators are built per split and each one asks for the preload.

    Without a guard the whole (trimmed) domain is decompressed two or three
    times over and two copies are held while it happens - about 3.3 GB each for
    GVZ. The guard is what makes --preload-precip usable, so it is worth a test:
    a second call must not touch the store.
    """
    from swafi.precip_archive import PrecipitationArchive

    class _CountingData:
        def __init__(self):
            self.computes = 0

        def compute(self):
            self.computes += 1
            return _Loaded()

    class _Loaded:
        sizes = {"time": 2, "y": 2, "x": 2}
        nbytes = 32

    archive = PrecipitationArchive.__new__(PrecipitationArchive)
    archive.full_grid_data = None
    archive.data = _CountingData()

    archive.preload_full_grid()
    assert archive.data.computes == 1
    first = archive.full_grid_data

    archive.preload_full_grid()
    assert archive.data.computes == 1, "the grid was recomputed on a second call"
    assert archive.full_grid_data is first
