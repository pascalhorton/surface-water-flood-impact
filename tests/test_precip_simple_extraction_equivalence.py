"""
A/B test: the optimized numpy implementation of the 'simple' event extraction
must reproduce the previous pandas implementation (kept verbatim below as the
reference) on identical inputs.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from swafi.precip import (SIMPLE_EVENT_HOURS_AFTER, SIMPLE_EVENT_HOURS_BEFORE,
                          Precipitation)


def _make_precip(values, freq, time_step):
    times = pd.date_range('2021-03-01', periods=len(values), freq=freq)
    ds = xr.Dataset(
        {'precip': (('time', 'y', 'x'), values[:, None, None].astype('float32'))},
        coords={'time': times, 'y': [1200500.0], 'x': [2500500.0]},
    )
    precip = object.__new__(Precipitation)
    precip.data = ds
    precip.time_step = time_step
    return precip


def _reference_simple(precip_obj, coords_row, strict_mode,
                      api_days_nb=30, api_reg=0.8):
    """The previous pandas implementation of the 'simple' method, verbatim."""
    time_series = precip_obj.data.sel(
        x=coords_row.x, y=coords_row.y).to_dataframe().reset_index()
    time_series['precip'] = time_series['precip'].astype('float64')
    dt = (time_series['time'].iloc[1]
          - time_series['time'].iloc[0]).total_seconds() / 3600
    window_minutes = [W for W in (5, 10, 20, 30) if W >= dt * 60]
    time_series['precip_q'] = time_series['precip'].rank(pct=True)

    threshold = time_series['precip'].quantile(0.98)
    exceed_times = time_series.loc[time_series['precip'] >= threshold, 'time']
    events = Precipitation._build_simple_event_dates(exceed_times, strict_mode)

    window_hours = [1, 2, 4, 6, 12, 24, 48, 72]
    for W in window_hours:
        n_steps = max(1, int(round(W / dt)))
        time_series[f'p_{W}h'] = time_series['precip'].rolling(n_steps).sum()
        time_series[f'p_{W}h_q'] = time_series[f'p_{W}h'].rank(pct=True)
    for W in window_minutes:
        n_steps = max(1, int(round(W / 60 / dt)))
        time_series[f'p_{W}min'] = time_series['precip'].rolling(n_steps).sum()
        time_series[f'p_{W}min_q'] = time_series[f'p_{W}min'].rank(pct=True)

    ts_indexed = time_series.set_index('time')
    time_idx = ts_indexed.index
    records = []
    for _, row in events.iterrows():
        if strict_mode:
            start = row['e_date']
            end = row['e_date'] + pd.Timedelta(hours=24)
        else:
            start = row['e_date'] - pd.Timedelta(hours=SIMPLE_EVENT_HOURS_BEFORE)
            end = row['e_date'] + pd.Timedelta(hours=SIMPLE_EVENT_HOURS_AFTER)
        i0 = time_idx.searchsorted(start, side='left')
        i1 = time_idx.searchsorted(end, side='right')
        window = ts_indexed.iloc[i0:i1]
        if window.empty:
            rec = {'i_max': np.nan, 'i_max_q': np.nan, 'i_max_date': pd.NaT}
            rec.update({f'p_{W}h': np.nan for W in window_hours})
            rec.update({f'p_{W}h_q': np.nan for W in window_hours})
            rec.update({f'p_{W}min': np.nan for W in window_minutes})
            rec.update({f'p_{W}min_q': np.nan for W in window_minutes})
        else:
            rec = {
                'i_max': window['precip'].max() / dt,
                'i_max_q': window['precip_q'].max(),
                'i_max_date': window['precip'].idxmax(),
            }
            for W in window_hours:
                rec[f'p_{W}h'] = window[f'p_{W}h'].max()
                rec[f'p_{W}h_q'] = window[f'p_{W}h_q'].max()
            for W in window_minutes:
                rec[f'p_{W}min'] = window[f'p_{W}min'].max()
                rec[f'p_{W}min_q'] = window[f'p_{W}min_q'].max()
        records.append(rec)
    events = pd.concat([events, pd.DataFrame(records, index=events.index)],
                       axis=1)
    events = events.astype({
        **{f'p_{W}h': 'float32' for W in window_hours},
        **{f'p_{W}h_q': 'float32' for W in window_hours},
        **{f'p_{W}min': 'float32' for W in window_minutes},
        **{f'p_{W}min_q': 'float32' for W in window_minutes},
    })

    daily_series = time_series.set_index('time').resample('D').agg(
        {'precip': 'sum'})
    daily_series['api'] = Precipitation._compute_api(
        daily_series['precip'].values, 24, api_days_nb, api_reg)
    daily_series['api_q'] = daily_series['api'].rank(pct=True)
    events = events.merge(
        daily_series.reset_index()[['time', 'api', 'api_q']],
        left_on='e_date', right_on='time', how='left').drop(columns=['time'])
    return events


def _run_new(precip_obj, coords_row, strict_mode):
    cell = precip_obj.data.sel(x=coords_row.x, y=coords_row.y)
    times = pd.DatetimeIndex(pd.to_datetime(cell['time'].values))
    values = np.asarray(cell['precip'].values, dtype='float64').reshape(-1)
    dt = (times[1] - times[0]).total_seconds() / 3600
    window_minutes = [W for W in (5, 10, 20, 30) if W >= dt * 60]
    # detection_window_h=None: the reference implements native-step detection
    return precip_obj._extract_events_simple(
        times, values, dt, window_minutes, strict_mode, 30, 0.8,
        detection_window_h=None)


def _synthetic_5min(days=60, with_nan=True):
    n = days * 288
    rng = np.random.default_rng(3)
    values = rng.gamma(0.08, 2.0, n)
    if with_nan:
        # Scattered NaN plus a short (3 h) gap. Kept short on purpose: the
        # reference implementation crashes on a fully-NaN event window (a
        # pre-existing pandas idxmax limitation), so equivalence is only
        # defined for windows containing at least one valid value.
        values[rng.choice(n, size=n // 200, replace=False)] = np.nan
        values[30 * 288 + 100:30 * 288 + 136] = np.nan
    return values


@pytest.mark.parametrize('strict', [True, False])
def test_simple_extraction_matches_reference_5min(strict):
    precip = _make_precip(_synthetic_5min(), '5min', 5 / 60)
    row = pd.Series({'cid': 1, 'x': 2500500.0, 'y': 1200500.0})

    ref = _reference_simple(precip, row, strict)
    new = _run_new(precip, row, strict)

    assert list(ref.columns) == list(new.columns)
    # check_dtype=False: the reference builds rows from dicts, which pandas 3
    # infers as datetime64[us]; the new code keeps the source ns resolution.
    pd.testing.assert_frame_equal(ref, new, check_exact=False,
                                  rtol=1e-5, atol=1e-8, check_dtype=False)


@pytest.mark.parametrize('strict', [True, False])
def test_simple_extraction_matches_reference_hourly(strict):
    n = 180 * 24
    rng = np.random.default_rng(9)
    values = rng.gamma(0.15, 1.5, n)
    precip = _make_precip(values, 'h', 1)
    row = pd.Series({'cid': 1, 'x': 2500500.0, 'y': 1200500.0})

    ref = _reference_simple(precip, row, strict)
    new = _run_new(precip, row, strict)

    assert list(ref.columns) == list(new.columns)
    # check_dtype=False: the reference builds rows from dicts, which pandas 3
    # infers as datetime64[us]; the new code keeps the source ns resolution.
    pd.testing.assert_frame_equal(ref, new, check_exact=False,
                                  rtol=1e-5, atol=1e-8, check_dtype=False)


def test_compute_api_fft_path_matches_direct():
    rng = np.random.default_rng(5)
    precip = rng.gamma(0.1, 2.0, 120_000)  # large enough for the FFT path
    kernel = np.power(0.8, np.arange(30 * 96) / 96)
    assert precip.size * kernel.size > 1e7

    api = Precipitation._compute_api(precip, 0.25, 30, 0.8)
    direct = np.concatenate(
        ([0.0], np.convolve(precip, kernel, mode='full')[:precip.size - 1]))
    np.testing.assert_allclose(api, direct, rtol=1e-9, atol=1e-9)
