"""
Two things in the event detection follow the resolution of the data rather
than the weather.

THE POSITIONS. The threshold is tested at every native time step, so the
number of chances a day gets to cross 10 mm scales with the resolution: 24 at
an hourly step, 288 at five minutes. A day whose 12-hour sum peaks at 9.8 mm
over hourly offsets can reach 10 mm at some 5-minute offset.

THE WINDOW. A window of w steps covers w * dt of wall clock, and with
end-labelled values and pandas' centring a 12-hour window sits at
(t-7h, t+5h] on hourly data against (t-6h05, t+5h55] at five minutes. Two
catalogues can test the same instant and still disagree on the sum there.

Phase Q measured the consequence on the real catalogues: each held 91.0% of
the 3,024 damaging days in their union, differing on borderline crossings
rather than on the rain.

detection_step_h pins both: detect as if the data had this time step. A
5-minute extraction with detection_step_h=1.0 asks the same question of the
same instants as an hourly one, and it is exactly a no-op on hourly data, so
switching it on does not move the hourly catalogue.
"""
import numpy as np
import pandas as pd
import pytest

from swafi.precip import Precipitation
from swafi.utils.event_extraction import detection_tag

WINDOW_H = 12
THRESHOLD_MM = 10
STEP_5MIN = 5 / 60


def _positions(freq, dt, step_h, periods=288 * 3):
    times = pd.DatetimeIndex(pd.date_range('2021-03-01', periods=periods,
                                           freq=freq))
    return times, Precipitation._detection_positions(times, dt, step_h)


def _extract(values, times, dt, step_h, threshold=THRESHOLD_MM,
             window_h=WINDOW_H, peak_days=True):
    precip = object.__new__(Precipitation)
    window_minutes = [W for W in (5, 10, 20, 30) if W >= dt * 60]
    events, _ = precip._extract_events_simple(
        pd.DatetimeIndex(times), np.asarray(values, dtype='float64'), dt,
        window_minutes, 30, 0.8, detection_window_h=window_h,
        detection_threshold=threshold, detection_centered=True,
        detection_peak_days=peak_days, detection_step_h=step_h)
    return events


def _dates(events):
    return set() if events is None else set(events['e_date'])


def _to_hourly(values, times):
    """Right-closed, right-labelled: the convention of both products."""
    hourly = (pd.Series(values, index=pd.DatetimeIndex(times))
              .resample('1h', closed='right', label='right').sum())
    return hourly.to_numpy(), pd.DatetimeIndex(hourly.index)


# --------------------------------------------------------- the positions ----

def test_no_detection_step_tests_every_native_step():
    _, mask = _positions('5min', STEP_5MIN, None)
    assert mask.all()


def test_a_detection_step_equal_to_the_data_step_is_a_no_op():
    _, mask = _positions('5min', STEP_5MIN, STEP_5MIN)
    assert mask.all()
    _, mask = _positions('1h', 1.0, 1.0)
    assert mask.all()


def test_an_hourly_detection_step_on_5min_data_keeps_the_whole_hours():
    times, mask = _positions('5min', STEP_5MIN, 1.0)
    assert mask.sum() == len(times) / 12
    assert (times[mask].minute == 0).all()
    assert set(times[mask].hour) == set(range(24))


def test_a_three_hourly_detection_step_is_anchored_at_midnight():
    times, mask = _positions('5min', STEP_5MIN, 3.0)
    kept = times[mask]
    assert (kept.minute == 0).all()
    assert set(kept.hour) == {0, 3, 6, 9, 12, 15, 18, 21}


def test_a_detection_step_finer_than_the_data_is_refused():
    with pytest.raises(ValueError, match="not a multiple"):
        _positions('1h', 1.0, STEP_5MIN)


def test_a_detection_step_off_the_data_grid_is_refused():
    # 90 minutes against an hourly step: half the grid it asks for does not
    # exist. It divides a day, so this is the step check and not that one.
    with pytest.raises(ValueError, match="not a multiple"):
        _positions('1h', 1.0, 1.5)


def test_a_detection_step_that_does_not_divide_a_day_is_refused():
    with pytest.raises(ValueError, match="does not divide a day"):
        _positions('1h', 1.0, 7.0)


def test_a_non_positive_detection_step_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        _positions('1h', 1.0, 0.0)


# ------------------------------------------------------------- the window ----

def test_the_window_spans_the_same_hours_at_both_resolutions():
    """(t-7h, t+5h] at an hourly step and at a 5-minute one."""
    lo_h, hi_h = Precipitation._detection_span(WINDOW_H, 1.0, True, 1.0)
    lo_f, hi_f = Precipitation._detection_span(WINDOW_H, STEP_5MIN, True, 1.0)

    # In hours, the covered interval is (lo - 1 step, hi].
    assert (lo_h - 1) * 1.0 == -7 and hi_h * 1.0 == 5
    assert np.isclose((lo_f - 1) * STEP_5MIN, -7)
    assert np.isclose(hi_f * STEP_5MIN, 5)
    # Both are twelve hours long.
    assert (hi_h - lo_h + 1) * 1.0 == WINDOW_H
    assert np.isclose((hi_f - lo_f + 1) * STEP_5MIN, WINDOW_H)


def test_without_a_detection_step_the_window_keeps_its_native_span():
    """The 5-minute window really is offset from the hourly one by 55 min.

    This is the second half of the problem, and the reason a stride on the
    positions alone would not have been enough.
    """
    lo_h, hi_h = Precipitation._detection_span(WINDOW_H, 1.0, True, None)
    lo_f, hi_f = Precipitation._detection_span(WINDOW_H, STEP_5MIN, True, None)
    assert ((lo_h - 1) * 1.0, hi_h * 1.0) == (-7, 5)
    assert np.isclose((lo_f - 1) * STEP_5MIN, -6 - 5 / 60)
    assert np.isclose(hi_f * STEP_5MIN, 5 + 55 / 60)


@pytest.mark.parametrize("dt,w", [(1.0, 12), (STEP_5MIN, 144), (1.0, 13)])
def test_the_native_span_is_pandas_centring(dt, w):
    lo, hi = Precipitation._detection_span(w * dt, dt, True, None)
    assert (lo, hi) == (-(w // 2), w - 1 - w // 2)


def test_a_trailing_window_ends_on_its_label():
    lo, hi = Precipitation._detection_span(WINDOW_H, 1.0, False, 1.0)
    assert (lo, hi) == (-11, 0)
    lo, hi = Precipitation._detection_span(WINDOW_H, STEP_5MIN, False, 1.0)
    assert (lo, hi) == (-143, 0)


@pytest.mark.parametrize("w", [2, 3, 12, 13, 144])
def test_the_summed_series_is_bit_identical_to_pandas_centring(w):
    """The accumulation is built from explicit offsets rather than center=True.

    pandas builds a centred window the same way, by shifting a trailing one,
    so the floats must agree exactly: anything else would move every event
    date in the existing catalogues.
    """
    rng = np.random.default_rng(0)
    values = rng.gamma(0.1, 2.0, 2000)
    lo, hi = -(w // 2), w - 1 - w // 2
    mine = Precipitation._detection_series(values, lo, hi)
    theirs = pd.Series(values).rolling(w, center=True).sum().to_numpy()
    assert np.array_equal(np.isnan(mine), np.isnan(theirs))
    assert np.array_equal(mine[~np.isnan(mine)], theirs[~np.isnan(theirs)])


# ----------------------------------------------- nothing already extracted ---

def test_hourly_data_is_untouched_by_an_hourly_detection_step():
    """The no-op the whole rollout rests on.

    DETECTION_STEP_H = 1.0 is switched on for every extraction, so if it
    moved a single hourly event the existing catalogue, the linked claims and
    every phase of results built on them would be invalidated.
    """
    rng = np.random.default_rng(17)
    n = 120 * 24
    values = rng.gamma(0.2, 2.0, n)
    times = pd.DatetimeIndex(pd.date_range('2021-03-01', periods=n, freq='1h'))

    without = _extract(values, times, 1.0, step_h=None)
    with_step = _extract(values, times, 1.0, step_h=1.0)

    assert without is not None and len(without) > 5
    pd.testing.assert_frame_equal(without, with_step)


def test_5min_data_is_untouched_by_a_5min_detection_step():
    rng = np.random.default_rng(19)
    n = 20 * 288
    values = rng.gamma(0.08, 2.0, n)
    times = pd.DatetimeIndex(pd.date_range('2021-03-01', periods=n,
                                           freq='5min'))

    without = _extract(values, times, STEP_5MIN, step_h=None)
    with_step = _extract(values, times, STEP_5MIN, step_h=STEP_5MIN)

    assert without is not None
    pd.testing.assert_frame_equal(without, with_step)


# ------------------------------------------------- the point of the change ---

def _borderline_series():
    """Six months of 5-minute rain whose 12-hour sums straddle 10 mm.

    One wet step in a hundred, 7 mm on average when wet, which puts a typical
    day's 12-hour maximum near the threshold. Rain that clears 10 mm
    comfortably is detected at any resolution and says nothing about the
    detection; it is the days sitting on the threshold that separate the
    variants, so the series is built to produce them.
    """
    rng = np.random.default_rng(23)
    n = 180 * 288
    times = pd.DatetimeIndex(pd.date_range('2021-01-01', periods=n,
                                           freq='5min'))
    wet = rng.random(n) < 0.01
    return rng.gamma(2.0, 3.5, n) * wet, times


def test_a_5min_extraction_reproduces_the_hourly_catalogue():
    """The requirement: one catalogue, whichever product detects it.

    The same rain at both resolutions, so any difference is the detection
    and nothing else. Dated on the exceedances rather than the intensity
    peak, because the peak of a 5-minute series and the peak of an hourly
    one are different instants by construction and that is a separate
    convention (see the test below).
    """
    values, times = _borderline_series()
    h_values, h_times = _to_hourly(values, times)

    hourly = _dates(_extract(h_values, h_times, 1.0, step_h=None,
                             peak_days=False))
    fine_native = _dates(_extract(values, times, STEP_5MIN, step_h=None,
                                  peak_days=False))
    fine_pinned = _dates(_extract(values, times, STEP_5MIN, step_h=1.0,
                                  peak_days=False))

    assert len(hourly) > 100, "the series should produce a usable catalogue"
    # Today: the fine step finds days the hourly step cannot, and never the
    # other way round - more positions can only add.
    assert fine_native > hourly
    # With the detection pinned: the same catalogue, exactly.
    assert fine_pinned == hourly


def test_peak_dating_is_the_only_residual_and_it_is_small():
    """With peak dating on, the two catalogues can still differ by a day.

    The event is dated on the intensity peak inside the window, and the peak
    of a 5-minute series need not fall in the same day as the peak of its
    hourly aggregate. The detection agrees; the dating convention is what is
    left, and it moves few days.
    """
    values, times = _borderline_series()
    h_values, h_times = _to_hourly(values, times)

    hourly = _dates(_extract(h_values, h_times, 1.0, step_h=None))
    fine_pinned = _dates(_extract(values, times, STEP_5MIN, step_h=1.0))

    union = hourly | fine_pinned
    agreement = len(hourly & fine_pinned) / len(union)
    # Phase Q's two real catalogues agreed on 82.0% of their union.
    assert agreement > 0.95, f"agreement fell to {agreement:.3f}"


# ----------------------------------------- the mechanism, day by known day ---

def _two_burst_series():
    """A day detected at a 5-minute offset and at no whole hour.

    Two bursts of 5.2 mm, 11 h 55 min apart, nothing else. A centred 12-hour
    window holds both only when centred exactly between them, which is one
    5-minute label and, by construction, not a whole hour:

        burst A   day 20, 00:10        the only exceeding label is 06:10
        burst B   day 20, 12:05        10.4 mm >= 10 mm there and nowhere else

    Aggregated to hours the bursts fall in the bins labelled 01:00 and 13:00,
    twelve hours apart, so no twelve-bin window holds both and the hourly
    series never crosses. The day is an artefact of the 5-minute sampling.
    """
    days = 30
    n = days * 288
    values = np.zeros(n)
    times = pd.DatetimeIndex(pd.date_range('2021-03-01', periods=n,
                                           freq='5min'))
    day = 20
    a = day * 288 + 2        # 00:10
    values[a] = 5.2
    values[a + 143] = 5.2    # 11 h 55 min later, 12:05
    assert times[a].strftime('%H:%M') == '00:10'
    assert times[a + 143].strftime('%H:%M') == '12:05'
    return values, times, times[a].normalize()


def test_the_detection_step_removes_a_day_only_the_fine_sampling_sees():
    values, times, day = _two_burst_series()

    native = _dates(_extract(values, times, STEP_5MIN, step_h=None))
    pinned = _dates(_extract(values, times, STEP_5MIN, step_h=1.0))
    h_values, h_times = _to_hourly(values, times)
    hourly = _dates(_extract(h_values, h_times, 1.0, step_h=None))

    # The rain is identical in all three; only the detection differs.
    assert np.isclose(values.sum(), h_values.sum())

    assert day in native, "the 5-minute sampling should catch this day"
    assert day not in hourly, "the hourly sampling cannot catch it"
    assert day not in pinned, "pinning the detection should bring them in line"
    assert pinned == hourly


def test_a_day_both_resolutions_see_is_untouched():
    values, times, day = _two_burst_series()
    # One burst well over the threshold: every window holding it exceeds, at
    # any offset and any resolution.
    values[:] = 0.0
    values[20 * 288 + 100] = 14.0

    native = _dates(_extract(values, times, STEP_5MIN, step_h=None))
    pinned = _dates(_extract(values, times, STEP_5MIN, step_h=1.0))
    h_values, h_times = _to_hourly(values, times)
    hourly = _dates(_extract(h_values, h_times, 1.0, step_h=None))

    assert day in native
    assert day in pinned
    assert day in hourly


def test_thirty_borderline_days_are_all_artefacts_of_the_fine_sampling():
    """The two-burst day repeated, to show it is not one lucky arrangement.

    Every second day carries a pair, so that no pair from one day can combine
    with a pair from the next (two bursts anywhere within 143 steps of each
    other sum to 10.4 mm and would exceed, which is how a first version of
    this test accidentally detected half the series). The offset within the
    day varies, so this is not one alignment either.

    A pair 11 h 55 min apart usually needs a window of twelve hours and five
    minutes to be caught, which only the 5-minute layout provides: an hourly
    window holding both would have to span thirteen bins. The exception is
    the offset that lands the second burst exactly on an hour boundary, which
    pulls it into the previous bin and leaves the pair eleven bins apart -
    three of these thirty days. Those three are genuine at both resolutions.

    What is asserted is not that the pinned catalogue is empty but that it is
    the hourly one, day for day, in a case built to be as awkward as the
    labelling allows.
    """
    days = 60
    n = days * 288
    values = np.zeros(n)
    times = pd.DatetimeIndex(pd.date_range('2021-03-01', periods=n,
                                           freq='5min'))
    for i, day in enumerate(range(0, days, 2)):
        a = day * 288 + (i % 12)   # a different offset each day
        values[a] = 5.2
        values[a + 143] = 5.2

    native = _dates(_extract(values, times, STEP_5MIN, step_h=None))
    pinned = _dates(_extract(values, times, STEP_5MIN, step_h=1.0))
    h_values, h_times = _to_hourly(values, times)
    hourly = _dates(_extract(h_values, h_times, 1.0, step_h=None))

    # Pinning the detection can only ever drop days, never add them.
    assert pinned <= native
    assert len(native) == 30, "every pair should be caught at the fine step"
    assert pinned == hourly
    assert len(pinned) == 3, "only the boundary offset survives"


def test_the_relative_threshold_follows_the_tested_positions():
    """With no absolute threshold the q98 is taken over the tested positions.

    Sampling one rolling window more often oversamples its peaks, so a q98
    read every five minutes is not the q98 read hourly. Taking it over the
    full series while testing a twelfth of it would reintroduce the
    resolution dependence.
    """
    rng = np.random.default_rng(5)
    n = 60 * 288
    values = rng.gamma(0.10, 2.0, n)
    times = pd.DatetimeIndex(pd.date_range('2021-03-01', periods=n,
                                           freq='5min'))

    lo, hi = Precipitation._detection_span(WINDOW_H, STEP_5MIN, True, 1.0)
    detection = Precipitation._detection_series(values, lo, hi)
    mask = Precipitation._detection_positions(times, STEP_5MIN, 1.0)
    on_grid = np.isfinite(detection) & mask
    q98_on_grid = float(np.quantile(detection[on_grid], 0.98))
    q98_full = float(np.quantile(detection[np.isfinite(detection)], 0.98))
    assert not np.isclose(q98_on_grid, q98_full)

    relative = _extract(values, times, STEP_5MIN, step_h=1.0, threshold=None)
    pinned = _extract(values, times, STEP_5MIN, step_h=1.0,
                      threshold=q98_on_grid)
    assert relative is not None
    pd.testing.assert_frame_equal(relative, pinned)


# ---------------------------------------------------------------- the tag ----

def test_the_tag_records_the_detection_step():
    base = detection_tag(WINDOW_H, THRESHOLD_MM, True, True, STEP_5MIN)
    pinned = detection_tag(WINDOW_H, THRESHOLD_MM, True, True, STEP_5MIN,
                           detection_step_h=1.0)
    assert pinned == base + "_step1h"


def test_the_tag_leaves_out_a_detection_step_that_changes_nothing():
    # On hourly data an hourly detection step is a no-op, so the extraction
    # keeps its name and is reused instead of re-run.
    assert (detection_tag(WINDOW_H, THRESHOLD_MM, True, True, 1.0,
                          detection_step_h=1.0)
            == detection_tag(WINDOW_H, THRESHOLD_MM, True, True, 1.0))


def test_the_tag_handles_a_sub_hourly_detection_step():
    tag = detection_tag(WINDOW_H, THRESHOLD_MM, True, True, STEP_5MIN,
                        detection_step_h=30 / 60)
    assert tag.endswith("_step30min")
