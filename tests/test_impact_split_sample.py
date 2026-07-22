"""
Tests for the train/validation/test split of the Impact base class.

The chronological mode holds out the last days of the period, so validation
measures what the model is asked to do (generalise to a later period) and the
probability threshold tuned on it transfers. The random modes interleave days
from the whole period between the splits.
"""
import numpy as np
import pandas as pd
import pytest

from swafi.impact import Impact


class _Options:
    min_nb_claims = 1
    random_state = 42


def _impact(n_days=400, seed=0):
    """A minimal Impact carrying only what split_sample needs."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2010-01-01', periods=n_days, freq='D')
    # A few events per day, to check that a day is never split.
    df = pd.DataFrame({
        'e_date': np.repeat(dates, 3),
        'cid': np.tile([1, 2, 3], n_days),
        'x': 1.0,
        'y': 2.0,
        'i_max_q': rng.random(3 * n_days),
        'target': (rng.random(3 * n_days) < 0.05).astype(int),
    })

    impact = object.__new__(Impact)
    impact.df = df
    impact.options = _Options()
    impact.features = ['i_max_q']
    impact.random_state = _Options.random_state
    impact.target_type = 'occurrence'
    return impact, dates


def _split_days(impact):
    """The set of days landing in each split, from the stored event properties."""
    return tuple(
        set(pd.to_datetime(pd.Series(list(ev[:, 0]))).dt.floor('D'))
        if len(ev) else set()
        for ev in (impact.events_train, impact.events_valid, impact.events_test)
    )


def test_chronological_holds_out_the_last_days():
    impact, dates = _impact()
    impact.split_sample(valid_test_size=0.25, test_size=0)

    train, valid, _ = _split_days(impact)
    assert max(train) < min(valid)  # strictly later, no interleaving
    assert len(valid) == pytest.approx(0.25 * len(dates), abs=1)
    assert train | valid == set(dates)


def test_chronological_with_a_test_split_is_ordered():
    impact, dates = _impact()
    impact.split_sample(valid_test_size=0.4, test_size=0.5)

    train, valid, test = _split_days(impact)
    assert max(train) < min(valid) < max(valid) < min(test)
    assert train | valid | test == set(dates)


def test_a_day_is_never_split_across_sets():
    """The events of one day share a storm across cells: splitting a day would
    leak it between training and validation."""
    for mode in ('chronological', 'random_days'):
        impact, _ = _impact()
        impact.split_sample(valid_test_size=0.25, test_size=0, split_mode=mode)
        train, valid, _ = _split_days(impact)
        assert not (train & valid), f"{mode} split a day across sets"


def test_random_days_interleaves_the_period():
    """The contrast with the chronological mode: random days share the period,
    so validation cannot see any drift between periods."""
    impact, _ = _impact()
    impact.split_sample(valid_test_size=0.25, test_size=0, split_mode='random_days')

    train, valid, _ = _split_days(impact)
    assert min(valid) < max(train)  # interleaved, not held out at the end


def test_split_is_reproducible_for_a_fixed_random_state():
    days = []
    for _ in range(2):
        impact, _ = _impact()
        impact.split_sample(valid_test_size=0.25, test_size=0,
                            split_mode='random_days')
        days.append(_split_days(impact)[1])
    assert days[0] == days[1]


def test_unknown_split_mode_is_rejected():
    impact, _ = _impact()
    with pytest.raises(ValueError, match="Unknown split mode"):
        impact.split_sample(split_mode='by_vibes')
