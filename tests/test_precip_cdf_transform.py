"""
Tests for the CDF (percentile) transform of the precipitation, which replaces
each value by its rank in the wet-step distribution of its own pixel.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from swafi.precip_archive import (
    CDF_MAX_LOG_EXCEEDANCE, PrecipitationArchive, get_cdf_levels)


def _archive(values, times=None):
    """Wrap a (time, y, x) array in an archive with the transforms available."""
    n_times = values.shape[0]
    if times is None:
        times = pd.date_range('2000-01-01', periods=n_times, freq='h')
    ys = np.arange(float(values.shape[1]))
    xs = np.arange(float(values.shape[2]))

    p = PrecipitationArchive.__new__(PrecipitationArchive)
    p.data = xr.Dataset(
        {'precip': (('time', 'y', 'x'), values)},
        coords={'time': times, 'y': ys, 'x': xs})
    p.precip_var = 'precip'
    p.time_axis_dim, p.y_axis_dim, p.x_axis_dim = 'time', 'y', 'x'
    p._transform_tag = ''
    p.cid_time_series = None
    p.full_grid_data = None

    return p


def _table(values, levels, wet_threshold=0.1):
    """The per-pixel value grid, computed directly (no caching or blocking)."""
    wet = np.where(values > wet_threshold, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        table = np.nanquantile(wet, levels, axis=0)
    return np.where(np.isfinite(table), table, np.inf)


def test_levels_return_period_spread():
    levels, step = get_cdf_levels('return_period', nb_levels=40)
    assert step == pytest.approx(CDF_MAX_LOG_EXCEEDANCE / 40)
    assert levels[0] == 0.0
    assert levels[-1] == pytest.approx(1 - 10 ** -(CDF_MAX_LOG_EXCEEDANCE - step))
    assert np.all(np.diff(levels) > 0)
    # The levels crowd towards 1: that is what keeps the extremes apart in the
    # output, which is evenly spaced by construction.
    assert levels[20] == pytest.approx(1 - 10 ** -2.0)


def test_levels_no_spread():
    levels, step = get_cdf_levels('none', nb_levels=40)
    assert step == pytest.approx(1 / 40)
    np.testing.assert_allclose(levels, np.arange(40) / 40)


def test_unknown_spread_raises():
    with pytest.raises(ValueError):
        get_cdf_levels('logit')


def test_dry_steps_map_to_zero_and_output_is_monotonic():
    rng = np.random.RandomState(0)
    values = rng.exponential(2.0, size=(4000, 1, 1))
    values[rng.rand(4000, 1, 1) < 0.9] = 0.0  # mostly dry
    levels, step = get_cdf_levels('return_period')

    p = _archive(values)
    p.cdf_transform(levels, _table(values, levels), step)
    out = p.data['precip'].to_numpy()

    assert out.min() == 0.0
    assert out.max() <= CDF_MAX_LOG_EXCEEDANCE
    np.testing.assert_array_equal(out[values == 0.0], 0.0)
    # Rank-preserving: sorting by input must sort the output.
    order = np.argsort(values.ravel())
    assert np.all(np.diff(out.ravel()[order]) >= 0)


def test_wet_values_use_the_full_output_range():
    """The point of ranking the wet steps only: with a percentile taken over all
    steps, a 90% dry series would pack every wet value into the top tenth."""
    rng = np.random.RandomState(1)
    values = rng.exponential(2.0, size=(4000, 1, 1))
    values[rng.rand(4000, 1, 1) < 0.9] = 0.0
    levels, step = get_cdf_levels('none')

    p = _archive(values)
    p.cdf_transform(levels, _table(values, levels), step)
    wet_out = p.data['precip'].to_numpy()[values > 0.1]

    assert wet_out.min() < 0.05
    assert wet_out.max() > 0.95
    # Roughly uniform over [0, 1] rather than crowded at the top.
    assert 0.4 < wet_out.mean() < 0.6


def test_pixels_with_different_climatologies_become_comparable():
    """Two pixels whose distributions differ by a factor 10 must give the same
    output for the same percentile: this is what the transform is for."""
    rng = np.random.RandomState(2)
    # Wet values start at 0.5 so that scaling a pixel cannot move any of them
    # across the wet threshold (which is absolute, hence not scale-free).
    series = rng.exponential(1.0, size=4000) + 0.5
    series[rng.rand(4000) < 0.85] = 0.0
    values = np.repeat(series[:, None, None], 2, axis=2)
    values[:, 0, 1] *= 10.0  # the second pixel is ten times wetter
    levels, step = get_cdf_levels('return_period')

    p = _archive(values)
    p.cdf_transform(levels, _table(values, levels), step)
    out = p.data['precip'].to_numpy()

    np.testing.assert_allclose(out[:, 0, 0], out[:, 0, 1], atol=1e-6)


def test_return_period_spread_separates_the_upper_tail():
    """The rarest wet values must land far apart, not be packed against 1."""
    values = np.concatenate([
        np.linspace(0.2, 10.0, 9990), np.array([50.0, 100.0, 500.0, 1000.0])
    ]).reshape(-1, 1, 1)
    levels, step = get_cdf_levels('return_period')

    p = _archive(values)
    p.cdf_transform(levels, _table(values, levels), step)
    out = p.data['precip'].to_numpy().ravel()

    median_out = out[np.argmin(np.abs(values.ravel() - 5.1))]
    assert median_out == pytest.approx(0.3, abs=0.15)  # ~ -log10(0.5)
    # The four rarest values keep a full unit of output between them and the bulk
    assert out[-1] - median_out > 2.5
    assert out[-1] > out[-4] > median_out


def test_table_marks_pixels_without_wet_steps():
    values = np.zeros((100, 1, 2))
    values[:, 0, 0] = 5.0  # only the first pixel is ever wet
    levels, _ = get_cdf_levels('none')
    table = _table(values, levels)

    assert np.isfinite(table[:, 0, 0]).all()
    assert np.isinf(table[:, 0, 1]).all()


def test_cdf_transform_is_idempotent():
    rng = np.random.RandomState(3)
    values = rng.exponential(2.0, size=(500, 1, 1))
    levels, step = get_cdf_levels('return_period')
    table = _table(values, levels)

    p = _archive(values)
    p.cdf_transform(levels, table, step)
    once = p.data['precip'].to_numpy().copy()
    p.cid_time_series = xr.DataArray([1.0], dims='cid')
    p.cdf_transform(levels, table, step)

    np.testing.assert_array_equal(p.data['precip'].to_numpy(), once)
    assert p.cid_time_series is not None
