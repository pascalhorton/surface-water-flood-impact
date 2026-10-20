"""The two-stage claim/event matching.

The second stage exists because ~9.6% of claims have a qualifying storm a day
or so outside the first window - noisy claim dates, not a detection failure.
About half of what it recovers is expected to be coincidence, so the point of
these tests is less that it finds more and more that what it finds stays
TELLABLE APART from the first stage's matches.
"""
import pandas as pd
import pytest

from swafi.damages import (Damages, SIMPLE_EVENT_HOURS_AFTER,
                           SIMPLE_EVENT_HOURS_BEFORE,
                           SIMPLE_EVENT_RELAXED_HOURS_AFTER,
                           SIMPLE_EVENT_RELAXED_HOURS_BEFORE)


def _claim(date='2015-07-10'):
    return pd.Series({'cid': 7, 'date_claim': pd.Timestamp(date)})


def _events_at(offsets_h, date='2015-07-10'):
    """One event per offset, in hours from the claim date at 0 h."""
    base = pd.Timestamp(date)
    df = pd.DataFrame({
        'eid': range(1, len(offsets_h) + 1),
        'cid': 7,
        'i_max_date': [base + pd.Timedelta(hours=h) for h in offsets_h],
        'i_max': [10.0] * len(offsets_h),
    }).sort_values('i_max_date').reset_index(drop=True)
    return {7: df}


def test_relaxed_window_contains_the_strict_one():
    """The design rests on this: stage 2 can only ever add, never move a match.

    If the relaxed window were not a superset, a claim matched at stage 1 could
    match a DIFFERENT event at stage 2, and 'match_step' would no longer mean
    'the strict window found nothing'.
    """
    assert SIMPLE_EVENT_RELAXED_HOURS_BEFORE >= SIMPLE_EVENT_HOURS_BEFORE
    assert SIMPLE_EVENT_RELAXED_HOURS_AFTER >= SIMPLE_EVENT_HOURS_AFTER


def test_relaxed_window_does_not_reach_forward():
    """The forward band was measured as coincidence, so it must stay excluded.

    A storm after the claim cannot have caused it. This is a deliberate
    asymmetry, not an oversight, and it is the kind of constant that gets
    'tidied' into symmetry by someone who has not read the measurement.
    """
    assert SIMPLE_EVENT_RELAXED_HOURS_AFTER == SIMPLE_EVENT_HOURS_AFTER


@pytest.mark.parametrize('offset_h', [-20.0, -31.0])
def test_second_stage_finds_the_backward_band(offset_h):
    claim, events = _claim(), _events_at([offset_h])
    assert Damages._get_potential_simple_events(claim, events) is None
    relaxed = Damages._get_potential_simple_events(
        claim, events,
        hours_before=SIMPLE_EVENT_RELAXED_HOURS_BEFORE,
        hours_after=SIMPLE_EVENT_RELAXED_HOURS_AFTER)
    assert relaxed is not None and len(relaxed) == 1


@pytest.mark.parametrize('offset_h', [-33.0, 30.0, 47.0])
def test_second_stage_still_rejects_what_is_outside_it(offset_h):
    """Relaxing is not opening. Beyond -32 h, and anywhere forward of +26 h,
    both stages must still decline."""
    claim, events = _claim(), _events_at([offset_h])
    for before, after in ((SIMPLE_EVENT_HOURS_BEFORE, SIMPLE_EVENT_HOURS_AFTER),
                          (SIMPLE_EVENT_RELAXED_HOURS_BEFORE,
                           SIMPLE_EVENT_RELAXED_HOURS_AFTER)):
        got = Damages._get_potential_simple_events(
            claim, events, hours_before=before, hours_after=after)
        assert got is None, f'{offset_h} h accepted by [-{before}, +{after}]'


def test_strict_match_is_unchanged_by_the_relaxed_window():
    """An event inside the strict window is found by stage 1, so stage 2 never
    runs for it - the default path must be bit-identical to before."""
    claim, events = _claim(), _events_at([18.0])
    strict = Damages._get_potential_simple_events(claim, events)
    assert strict is not None and list(strict['eid']) == [1]


def test_default_window_is_the_published_one():
    """Slide 8's distribution is what justifies [-8, +26]; changing it silently
    would invalidate every linked sample measured against it."""
    assert (SIMPLE_EVENT_HOURS_BEFORE, SIMPLE_EVENT_HOURS_AFTER) == (8, 26)


class _FakeDamages:
    """Just enough of Damages for set_target_values_from_damages()."""

    def __init__(self, claims):
        self.claims = claims


def _linked_claims(rows):
    return pd.DataFrame(rows, columns=['date_claim', 'eid', 'selection',
                                       'target', 'match_step'])


def test_match_step_reaches_the_events_file():
    from swafi.events import Events
    ev = Events()
    ev.events = pd.DataFrame({'eid': [1, 2, 3], 'cid': [7, 7, 7]})
    ev.set_target_values_from_damages(_FakeDamages(_linked_claims([
        (pd.Timestamp('2015-07-10'), 1, 1, 1, 1),
        (pd.Timestamp('2015-07-11'), 2, 1, 1, 2),
    ])))
    by_eid = dict(zip(ev.events['eid'], ev.events['match_step']))
    assert by_eid[1] == 1, 'strict match lost its stage'
    assert by_eid[2] == 2, 'relaxed match lost its stage'
    assert by_eid[3] == 0, 'a negative must carry 0, not NaN'


def test_an_event_matched_both_ways_counts_as_strict():
    """One event can be the strict match of one claim and the relaxed match of
    another. It would be a positive with the relaxed stage off, so dropping
    match_step == 2 must not remove it."""
    from swafi.events import Events
    ev = Events()
    ev.events = pd.DataFrame({'eid': [1], 'cid': [7]})
    ev.set_target_values_from_damages(_FakeDamages(_linked_claims([
        (pd.Timestamp('2015-07-10'), 1, 1, 1, 2),
        (pd.Timestamp('2015-07-12'), 1, 1, 1, 1),
    ])))
    assert ev.events['match_step'].iloc[0] == 1


def test_linkage_without_the_column_still_works():
    """Samples linked before this change have no match_step; reading them back
    must not break."""
    from swafi.events import Events
    ev = Events()
    ev.events = pd.DataFrame({'eid': [1, 2], 'cid': [7, 7]})
    claims = _linked_claims([(pd.Timestamp('2015-07-10'), 1, 1, 1, 1)])
    ev.set_target_values_from_damages(_FakeDamages(claims.drop(columns='match_step')))
    assert 'match_step' not in ev.events.columns
    assert ev.events['target'].tolist() == [1, 0]


def test_match_step_is_never_a_default_feature():
    """It is 0 for every negative, so a model given it would score perfectly
    and learn nothing. The default feature lists are built by name in
    Impact._define_potential_features; this pins that match_step is not among
    the names it can ever add.
    """
    import inspect
    import swafi.impact as impact_module

    src = inspect.getsource(impact_module)
    # The feature lists are literal names in the source, so a new entry would
    # have to appear here. Catches the column being added to a list by hand.
    i = src.index("self.tabular_features = {}")
    assert "match_step" not in src[i:], (
        "match_step appears in the feature definitions - it is a perfect "
        "leak and must not be selectable")
