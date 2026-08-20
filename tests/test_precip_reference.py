"""
Tests for the per-cell training reference used by the simple event extraction
to keep the *_q normalisation consistent across the train/test boundary.
"""

import numpy as np
import pandas as pd
import pytest

from swafi.precip import Precipitation
from swafi.utils.precip_reference import build_cdf, cdf_percentile


def _series(years, seed, scale):
    times = pd.date_range(f'{years[0]}-01-01', f'{years[1]}-12-31 23:00', freq='h')
    rng = np.random.RandomState(seed)
    x = rng.exponential(scale, len(times))
    x[rng.rand(len(times)) < 0.85] = 0.0  # mostly dry
    return pd.DatetimeIndex(times), x


def _extract(precip_obj, times, values, ref=None, build_ref=False):
    return precip_obj._extract_events_simple(
        times, values, 1.0, [], 30, 0.8, 1.0, ref=ref, build_ref=build_ref)


def test_cdf_roundtrip_matches_empirical_rank():
    rng = np.random.RandomState(0)
    sample = rng.exponential(1.0, 20000)
    cdf = build_cdf(sample)
    query = np.array([0.5, 1.0, 2.0, 5.0])
    approx = cdf_percentile(cdf, query)
    exact = np.searchsorted(np.sort(sample), query, side='right') / sample.size
    assert np.max(np.abs(approx - exact)) < 0.01


def test_cdf_percentile_handles_nan_and_bounds():
    cdf = build_cdf(np.arange(100.0))
    out = cdf_percentile(cdf, np.array([np.nan, -10.0, 1e6]))
    assert np.isnan(out[0])
    assert out[1] == 0.0  # below the reference
    assert out[2] == 1.0  # above the reference


def test_build_ref_does_not_change_events():
    """Passing build_ref=True must not alter the extracted features (only emit
    the reference in addition)."""
    times, values = _series((2005, 2010), seed=1, scale=1.0)
    p = Precipitation.__new__(Precipitation)
    ev_plain, ref_none = _extract(p, times, values, build_ref=False)
    ev_built, ref_out = _extract(p, times, values, build_ref=True)
    assert ref_none is None
    assert ref_out is not None and 'q98' in ref_out
    pd.testing.assert_frame_equal(ev_plain, ev_built)


def test_applied_reference_ranks_against_training():
    """Test events ranked against a training reference must match ranking the
    same values against the training distribution (not the test period)."""
    p = Precipitation.__new__(Precipitation)
    t_tr, p_tr = _series((2005, 2016), seed=1, scale=1.0)
    t_te, p_te = _series((2017, 2019), seed=2, scale=1.6)  # wetter test period

    _, ref = _extract(p, t_tr, p_tr, build_ref=True)
    ev_ref, _ = _extract(p, t_te, p_te, ref=ref)
    ev_self, _ = _extract(p, t_te, p_te)

    # The detection threshold stored in the reference is the training q98.
    assert ref['q98'] == pytest.approx(
        float(np.quantile(p_tr[np.isfinite(p_tr)], 0.98)))

    # i_max_q against the reference matches the exact rank vs the training precip.
    sorted_tr = np.sort(p_tr[~np.isnan(p_tr)])
    exact = p._pct_rank(sorted_tr, ev_ref['i_max'].to_numpy())  # dt=1 -> v_max==i_max
    assert np.nanmax(np.abs(exact - ev_ref['i_max_q'].to_numpy())) < 0.02

    # And it genuinely differs from self-normalisation.
    assert not np.allclose(np.nanmedian(ev_ref['i_max_q']),
                           np.nanmedian(ev_self['i_max_q']), atol=1e-6)
