"""
The 'simple' event extraction detects exceedances on a configurable
accumulation window (default 1 h): on 5-min data the default flags sustained
hourly intensity, while detection_window_h=None flags native-step bursts.
On hourly data the default is equivalent to the native time step.

The threshold on that accumulation is the per-cell q98 by default, or an
absolute value [mm] when detection_threshold is given (e.g. p_12h >= 10mm).

The window is centred on the step it labels (detection_centered) and the
events are dated on the intensity peak of each exceeding window
(detection_peak_days). Both only bear on a window spanning several steps; the
helper below turns them off so each test enables what it exercises.
"""
import numpy as np
import pandas as pd

from swafi.precip import Precipitation


def _extract(values, freq, dt, detection_window_h, detection_threshold=None,
             detection_centered=False, detection_peak_days=False):
    times = pd.date_range('2021-03-01', periods=len(values), freq=freq)
    precip = object.__new__(Precipitation)
    window_minutes = [W for W in (5, 10, 20, 30) if W >= dt * 60]
    events, _ = precip._extract_events_simple(
        pd.DatetimeIndex(times), values.astype('float64'), dt, window_minutes,
        30, 0.8, detection_window_h=detection_window_h,
        detection_threshold=detection_threshold,
        detection_centered=detection_centered,
        detection_peak_days=detection_peak_days)
    return events


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


def test_absolute_threshold_selects_the_exceedance_days():
    """detection_window_h=12 with detection_threshold=10 must select exactly
    the days where the rolling 12 h accumulation reaches 10 mm."""
    n = 300 * 24
    rng = np.random.default_rng(13)
    values = rng.gamma(0.15, 1.5, n)
    times = pd.date_range('2021-03-01', periods=n, freq='h')

    events = _extract(values, 'h', 1, detection_window_h=12,
                      detection_threshold=10)

    p_12h = pd.Series(values).rolling(12).sum().to_numpy()
    expected = pd.DatetimeIndex(times[p_12h >= 10]).floor('D').unique()
    assert set(events['e_date']) == set(expected)
    assert len(events) > 0


def test_absolute_threshold_is_period_independent():
    """Unlike q98, an absolute threshold selects on the values themselves: a
    drier period yields fewer events instead of the same 2% of the steps."""
    n = 300 * 24
    rng = np.random.default_rng(17)
    wet = rng.gamma(0.15, 1.5, n)
    dry = wet / 3

    abs_wet = _extract(wet, 'h', 1, 12, detection_threshold=10)
    abs_dry = _extract(dry, 'h', 1, 12, detection_threshold=10)
    q98_wet = _extract(wet, 'h', 1, 12)
    q98_dry = _extract(dry, 'h', 1, 12)

    # A cell that never reaches the threshold yields no event at all (None).
    assert abs_dry is None or len(abs_dry) < len(abs_wet)
    assert len(abs_wet) > 0
    # Scaling the series leaves the quantile-based detection untouched.
    assert set(q98_dry['e_date']) == set(q98_wet['e_date'])


def _storms(n_days, storms):
    """Dry hourly series with single-hour storms at the given (day, hour, mm)."""
    values = np.zeros(n_days * 24)
    for day, hour, amount in storms:
        values[day * 24 + hour] = amount
    return values


_START = pd.Timestamp('2021-03-01')


def test_peak_days_collapses_a_storm_straddling_midnight():
    """A trailing 12 h window keeps a 20:00 storm above the threshold until
    07:00 the next day, flagging two days for one storm. All those windows
    peak on the same hour, so dating on the peak gives back a single event."""
    values = _storms(30, [(20, 20, 15.0)])

    split = _extract(values, 'h', 1, 12, detection_threshold=10)
    peaked = _extract(values, 'h', 1, 12, detection_threshold=10,
                      detection_peak_days=True)

    assert set(split['e_date']) == {_START + pd.Timedelta(days=20),
                                    _START + pd.Timedelta(days=21)}
    assert set(peaked['e_date']) == {_START + pd.Timedelta(days=20)}


def test_peak_days_keeps_both_days_when_both_peaks_stand_alone():
    """Two peaks on consecutive days, each crossing the threshold on its own,
    keep one event each: dating on the peak deduplicates a storm without
    flattening a spell that genuinely spans several days."""
    values = _storms(30, [(10, 22, 12.0), (11, 3, 30.0)])

    peaked = _extract(values, 'h', 1, 12, detection_threshold=10,
                      detection_peak_days=True)

    assert set(peaked['e_date']) == {_START + pd.Timedelta(days=10),
                                     _START + pd.Timedelta(days=11)}


def test_peak_days_drops_a_shoulder_that_does_not_stand_alone():
    """A shoulder too weak to carry a window over the threshold on its own is
    never the peak of one, so it adds no event day."""
    values = _storms(30, [(10, 22, 6.0), (11, 3, 30.0)])

    peaked = _extract(values, 'h', 1, 12, detection_threshold=10,
                      detection_peak_days=True)

    assert set(peaked['e_date']) == {_START + pd.Timedelta(days=11)}


def test_peak_days_keeps_storms_separated_by_a_lull_distinct():
    """Windows over distinct storms peak on distinct hours, so dating on the
    peak never fuses two storms separated by a dry spell."""
    values = _storms(30, [(10, 20, 15.0), (13, 20, 15.0)])

    split = _extract(values, 'h', 1, 12, detection_threshold=10)
    peaked = _extract(values, 'h', 1, 12, detection_threshold=10,
                      detection_peak_days=True)

    assert len(split) == 4  # each storm flags its day and the next
    assert set(peaked['e_date']) == {_START + pd.Timedelta(days=10),
                                     _START + pd.Timedelta(days=13)}


def test_peak_days_labels_the_rain_day_not_the_crossing_day():
    """With a trailing window the accumulation can cross the threshold only
    after midnight, while the rain responsible for it fell the day before.
    Dating on the peak of the window puts the event on the rain day, which
    dating on the exceedance gets wrong."""
    values = _storms(30, [(10, 22, 6.0), (10, 23, 3.0), (11, 0, 2.0)])

    split = _extract(values, 'h', 1, 12, detection_threshold=10)
    peaked = _extract(values, 'h', 1, 12, detection_threshold=10,
                      detection_peak_days=True)

    assert set(split['e_date']) == {_START + pd.Timedelta(days=11)}
    assert set(peaked['e_date']) == {_START + pd.Timedelta(days=10)}


def test_peak_days_reduces_the_event_count():
    """On a realistic series, peak dating removes the days a storm was counted
    on more than once. Every kept day is a detected day or the day before one
    (with a trailing window the peak precedes the exceedance it caused)."""
    n = 300 * 24
    rng = np.random.default_rng(23)
    values = rng.gamma(0.15, 1.5, n)

    split = _extract(values, 'h', 1, 12, detection_threshold=10)
    peaked = _extract(values, 'h', 1, 12, detection_threshold=10,
                      detection_peak_days=True)

    assert len(peaked) < len(split)
    detected = set(split['e_date'])
    allowed = detected | {d - pd.Timedelta(days=1) for d in detected}
    assert set(peaked['e_date']) <= allowed


def test_peak_days_keeps_one_event_per_peak_of_a_long_spell():
    """A spell long enough that its windows peak on different days keeps one
    event per peak day: the rule adapts instead of collapsing to one event."""
    # Three daily peaks, 24 h apart, each strong enough to carry its own
    # 12 h window over the threshold.
    values = _storms(30, [(10, 12, 12.0), (11, 12, 20.0), (12, 12, 15.0)])

    peaked = _extract(values, 'h', 1, 12, detection_threshold=10,
                      detection_peak_days=True)

    assert set(peaked['e_date']) == {_START + pd.Timedelta(days=d)
                                     for d in (10, 11, 12)}


def test_centered_window_flags_the_steps_around_the_rain():
    """A centred window flags the steps the storm falls around; a trailing one
    only flags those after it, lagging the storm by up to the window length."""
    values = _storms(30, [(10, 12, 15.0)])
    peak = _START + pd.Timedelta(days=10, hours=12)
    times = pd.DatetimeIndex(pd.date_range('2021-03-01', periods=len(values),
                                           freq='h'))

    def flagged(centered):
        detection = pd.Series(values).rolling(
            12, center=centered).sum().to_numpy()
        return times[detection >= 10]

    assert flagged(True).min() < peak < flagged(True).max()
    assert flagged(False).min() == peak  # never before the rain


def test_centered_window_is_a_noop_on_a_single_step_window():
    """Both options only bear on a multi-step window, so the hourly default
    (a 1 h window on hourly data) is unaffected by either."""
    n = 120 * 24
    rng = np.random.default_rng(29)
    values = rng.gamma(0.15, 1.5, n)

    plain = _extract(values, 'h', 1, 1)
    both = _extract(values, 'h', 1, 1, detection_centered=True,
                    detection_peak_days=True)
    pd.testing.assert_frame_equal(plain, both)


def test_absolute_threshold_ignores_the_reference():
    """The reference normalises the *_q features, but an absolute detection
    threshold is already period-independent and must not be overridden."""
    n = 300 * 24
    rng = np.random.default_rng(19)
    values = rng.gamma(0.15, 1.5, n)
    times = pd.DatetimeIndex(pd.date_range('2021-03-01', periods=n, freq='h'))
    precip = object.__new__(Precipitation)

    _, ref = precip._extract_events_simple(
        times, values, 1, [], 30, 0.8, detection_window_h=12,
        detection_centered=False, detection_peak_days=False, build_ref=True)
    # The stored threshold is the q98 of the 12 h accumulation, not 10 mm.
    assert ref['q98'] != 10

    with_ref, _ = precip._extract_events_simple(
        times, values, 1, [], 30, 0.8, detection_window_h=12,
        detection_centered=False, detection_peak_days=False,
        detection_threshold=10, ref=ref)
    without_ref = _extract(values, 'h', 1, 12, detection_threshold=10)
    assert set(with_ref['e_date']) == set(without_ref['e_date'])
