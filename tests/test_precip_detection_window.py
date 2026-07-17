"""
The 'simple' event extraction detects exceedances on a configurable
accumulation window (default 1 h): on 5-min data the default flags sustained
hourly intensity, while detection_window_h=None flags native-step bursts.
On hourly data the default is equivalent to the native time step.
"""
import numpy as np
import pandas as pd

from swafi.precip import Precipitation


def _extract(values, freq, dt, detection_window_h):
    times = pd.date_range('2021-03-01', periods=len(values), freq=freq)
    precip = object.__new__(Precipitation)
    window_minutes = [W for W in (5, 10, 20, 30) if W >= dt * 60]
    return precip._extract_events_simple(
        pd.DatetimeIndex(times), values.astype('float64'), dt, window_minutes,
        True, 30, 0.8, detection_window_h=detection_window_h)


def test_detection_window_5min():
    days = 60
    n = days * 288
    rng = np.random.default_rng(7)
    values = rng.gamma(0.08, 2.0, n)

    # Thresholds of the unmodified background (the engineered days below only
    # zero out ~3% of the series, so the actual thresholds are marginally
    # lower; the 0.9 safety factor covers that shift).
    thr_native = np.quantile(values, 0.98)
    sums_1h = pd.Series(values).rolling(12).sum().to_numpy()
    thr_1h = np.quantile(sums_1h[np.isfinite(sums_1h)], 0.98)

    # Spike day: a single 5-min burst above the native threshold whose hourly
    # accumulation stays below the hourly threshold. The previous hour is
    # zeroed too so no window labelled on the spike day reaches back into
    # background rain.
    spike_day = 20
    spike_value = 1.5 * thr_native
    assert spike_value < 0.9 * thr_1h
    values[spike_day * 288 - 12:(spike_day + 1) * 288] = 0.0
    values[spike_day * 288 + 144] = spike_value  # at 12:00

    # Sustained day: one hour of moderate rain below the native threshold but
    # above the hourly threshold once accumulated.
    sustained_day = 40
    sustained_value = 0.8 * thr_native
    assert 12 * sustained_value > thr_1h
    values[sustained_day * 288 - 12:(sustained_day + 1) * 288] = 0.0
    values[sustained_day * 288 + 144:sustained_day * 288 + 156] = sustained_value

    start = pd.Timestamp('2021-03-01')
    spike_date = start + pd.Timedelta(days=spike_day)
    sustained_date = start + pd.Timedelta(days=sustained_day)

    native = _extract(values, '5min', 5 / 60, detection_window_h=None)
    hourly = _extract(values, '5min', 5 / 60, detection_window_h=1)

    assert spike_date in set(native['e_date'])
    assert sustained_date not in set(native['e_date'])
    assert sustained_date in set(hourly['e_date'])
    assert spike_date not in set(hourly['e_date'])


def test_detection_window_default_is_noop_on_hourly():
    n = 120 * 24
    rng = np.random.default_rng(11)
    values = rng.gamma(0.15, 1.5, n)

    default = _extract(values, 'h', 1, detection_window_h=1)
    native = _extract(values, 'h', 1, detection_window_h=None)
    pd.testing.assert_frame_equal(default, native)
