import numpy as np
import pandas as pd
import pytest
import xarray as xr

from swafi.precip import Precipitation


def _make_precip(values, freq, time_step):
    """Build a minimal Precipitation object around a single-cell time series."""
    times = pd.date_range('2021-06-01', periods=len(values), freq=freq)
    ds = xr.Dataset(
        {'precip': (('time', 'y', 'x'), values[:, None, None].astype('float32'))},
        coords={'time': times, 'y': [1200500.0], 'x': [2500500.0]},
    )
    precip = object.__new__(Precipitation)
    precip.data = ds
    precip.time_step = time_step
    return precip


def _coords_row():
    return pd.Series({'cid': 1, 'x': 2500500.0, 'y': 1200500.0})


def _five_min_series_with_burst():
    """20 days of 5-min noise with a [1, 2, 4, 8, 2, 1] mm burst on day 10, 12:00."""
    n = 20 * 288
    rng = np.random.default_rng(42)
    values = rng.uniform(0.0, 0.01, n)
    burst_start = 10 * 288 + 12 * 12  # day 10, 12:00
    values[burst_start:burst_start + 6] = [1.0, 2.0, 4.0, 8.0, 2.0, 1.0]
    return values


def test_simple_method_minute_windows_from_5min_data():
    precip = _make_precip(_five_min_series_with_burst(), '5min', 5 / 60)

    events = precip._extract_events(_coords_row(), method='simple')

    for col in ('p_5min', 'p_10min', 'p_20min', 'p_30min'):
        assert col in events.columns
        assert f'{col}_q' in events.columns

    # The windows must not collapse to a single value (the old hourly bug)
    burst = events[events.e_date == pd.Timestamp('2021-06-11')].iloc[0]
    assert burst.p_5min == pytest.approx(8.0, abs=0.1)
    assert burst.p_10min == pytest.approx(12.0, abs=0.1)
    assert burst.p_20min == pytest.approx(16.0, abs=0.1)
    assert burst.p_30min == pytest.approx(18.0, abs=0.1)

    # Longer windows accumulate at least as much precipitation
    valid = events.dropna(subset=['p_5min', 'p_10min', 'p_20min', 'p_30min'])
    assert (valid.p_5min <= valid.p_10min + 1e-6).all()
    assert (valid.p_10min <= valid.p_20min + 1e-6).all()
    assert (valid.p_20min <= valid.p_30min + 1e-6).all()

    # i_max is an intensity in mm/h: 8 mm in 5 min -> 96 mm/h
    assert burst.i_max == pytest.approx(96.0, abs=1.0)


def test_simple_method_skips_minute_windows_for_hourly_data():
    n = 20 * 24
    rng = np.random.default_rng(42)
    values = rng.uniform(0.0, 0.01, n)
    values[10 * 24 + 12] = 10.0  # day 10, 12:00
    precip = _make_precip(values, 'h', 1)

    events = precip._extract_events(_coords_row(), method='simple')

    for col in ('p_5min', 'p_10min', 'p_20min', 'p_30min'):
        assert col not in events.columns
        assert f'{col}_q' not in events.columns
    assert 'p_1h' in events.columns

    # i_max stays in mm/h (dt = 1 h -> unchanged)
    burst = events[events.e_date == pd.Timestamp('2021-06-11')].iloc[0]
    assert burst.i_max == pytest.approx(10.0, abs=0.1)


def test_classic_method_skips_minute_windows_for_hourly_data():
    n = 20 * 24
    values = np.zeros(n)
    values[10 * 24:11 * 24] = 1.0  # 24 h of 1 mm/h -> p_sum = 24 >= 10
    precip = _make_precip(values, 'h', 1)

    events = precip._extract_events(_coords_row(), method='classic')

    assert len(events) == 1
    for col in ('p_5min', 'p_10min', 'p_20min', 'p_30min'):
        assert col not in events.columns
    assert events.iloc[0].i_max == pytest.approx(1.0)
    assert events.iloc[0].i_mean == pytest.approx(1.0)
    assert events.iloc[0].p_sum == pytest.approx(24.0)


def test_classic_method_minute_windows_from_5min_data():
    values = np.zeros(20 * 288)
    burst_start = 10 * 288 + 12 * 12
    values[burst_start:burst_start + 6] = [1.0, 2.0, 4.0, 8.0, 2.0, 1.0]  # 18 mm
    precip = _make_precip(values, '5min', 5 / 60)

    events = precip._extract_events(_coords_row(), method='classic')

    assert len(events) == 1
    event = events.iloc[0]
    assert event.p_sum == pytest.approx(18.0)
    assert event.p_5min == pytest.approx(8.0)
    assert event.p_10min == pytest.approx(12.0)
    assert event.p_20min == pytest.approx(16.0)
    assert event.p_30min == pytest.approx(18.0)
    # Intensities in mm/h regardless of the native step
    assert event.i_max == pytest.approx(96.0)
    assert event.i_mean == pytest.approx(3.0 * 12)
