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


# ---------------------------------------------------------------------------
# Switching the relaxed positives back off at training time
# ---------------------------------------------------------------------------

def _sample(match_steps, targets=None, nb_claims=None):
    """A minimal linked sample: one event per match_step given."""
    from swafi.events import Events
    n = len(match_steps)
    if targets is None:
        targets = [1 if s else 0 for s in match_steps]
    if nb_claims is None:
        nb_claims = [1 if s else 0 for s in match_steps]
    ev = Events()
    ev.events = pd.DataFrame({
        'eid': range(1, n + 1),
        'cid': 7,
        'target': [float(t) for t in targets],
        'nb_claims': [float(c) for c in nb_claims],
        'match_step': match_steps,
    })
    return ev


def test_dropping_relaxed_positives_demotes_only_stage_two():
    ev = _sample([1, 2, 0, 2, 1])
    assert ev.drop_relaxed_positives() == 2
    assert ev.events['target'].tolist() == [1, 0, 0, 0, 1]
    assert ev.events['match_step'].tolist() == [1, 0, 0, 0, 1]


def test_demoted_positives_stay_in_the_sample_as_negatives():
    """The whole point: a strict linkage keeps that storm as a NEGATIVE, so
    zeroing the label is right and dropping the row is not - the two samples
    must differ by their labels alone."""
    ev = _sample([1, 2])
    n_before = len(ev.events)
    ev.drop_relaxed_positives()
    assert len(ev.events) == n_before
    assert (ev.events['target'] == 0).sum() == 1


def test_demoted_positives_survive_the_min_nb_claims_filter():
    """nb_claims is zeroed, and Impact spares nb_claims == 0 explicitly. If it
    were left at 1 a --min-nb-claims 2 run would drop the row instead of
    keeping it as a negative, and the comparison would be against a different
    sample rather than different labels."""
    import inspect
    import swafi.impact as impact_module

    ev = _sample([2])
    ev.drop_relaxed_positives()
    assert ev.events['nb_claims'].tolist() == [0.0]

    src = inspect.getsource(impact_module)
    assert "self.df['nb_claims'] == 0" in src, (
        "Impact no longer spares nb_claims == 0 from the min_nb_claims "
        "filter; demoting a positive by zeroing nb_claims would now drop it "
        "from the sample instead of keeping it as a negative")


def test_dropping_without_the_column_fails_loudly():
    """A sample linked with relaxed matching off has nothing to drop. Silently
    doing nothing would make the two legs of the experiment identical and the
    result meaningless."""
    ev = _sample([1])
    ev.events = ev.events.drop(columns='match_step')
    with pytest.raises(ValueError, match='match_step'):
        ev.drop_relaxed_positives()


def test_remove_duplicates_keeps_the_strict_stage():
    """The kept row is the one with most claims, which need not be the strict
    one. If the group's stage came from that row alone, a positive the strict
    linkage would also have produced could end up marked relaxed - and then
    dropped."""
    from swafi.events import Events
    ev = Events()
    ev.events = pd.DataFrame({
        'eid': [1, 2],
        'cid': [7, 7],
        'i_max_date': [pd.Timestamp('2015-07-10 18:00')] * 2,
        'target': [1.0, 1.0],
        'nb_claims': [3.0, 1.0],
        'match_step': [2, 1],
    })
    ev.remove_duplicates()
    assert len(ev.events) == 1
    assert ev.events['match_step'].iloc[0] == 1
    assert ev.events['nb_claims'].iloc[0] == 4.0


def test_remove_duplicates_does_not_collapse_a_positive_to_zero():
    """Masking the zeros matters: a group of one positive and several
    negatives must keep the positive's stage, since 0 means 'not a
    positive'."""
    from swafi.events import Events
    ev = Events()
    ev.events = pd.DataFrame({
        'eid': [1, 2],
        'cid': [7, 7],
        'i_max_date': [pd.Timestamp('2015-07-10 18:00')] * 2,
        'target': [1.0, 0.0],
        'nb_claims': [2.0, 0.0],
        'match_step': [2, 0],
    })
    ev.remove_duplicates()
    assert ev.events['match_step'].iloc[0] == 2, (
        'the surviving positive lost its stage and can no longer be dropped')


def test_every_training_script_honours_the_option():
    """A flag that works in some scripts and silently does nothing in others
    is worse than no flag: the run would look like the experiment and not be
    it. This pins the wiring rather than trusting it."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / 'scripts' / 'impact_functions'
    scripts = sorted(root.glob('train_*_occurrence.py'))
    assert scripts, 'no training scripts found - has the layout moved?'
    for path in scripts:
        src = path.read_text(encoding='utf-8')
        assert 'options.exclude_relaxed_positives' in src, (
            f'{path.name} does not honour --exclude-relaxed-positives')
        assert 'drop_relaxed_positives()' in src, (
            f'{path.name} reads the option but never acts on it')


def test_the_relax_tag_reaches_the_events_sample_name():
    """The relaxed sample must not be able to overwrite the strict one: the
    label that names the events file has to carry the window, exactly as the
    intermediate linked-damages pickle does. This was a real defect - the tag
    was on the pickle only."""
    from pathlib import Path

    path = (Path(__file__).resolve().parents[1] / 'scripts' /
            'link_claims_events' / 'compute_claims_events_link.py')
    src = path.read_text(encoding='utf-8')
    i = src.index('LABEL_RESULTING_FILE = ')
    tail = src[i:src.index('\n\n', i)]
    assert '_RELAX_TAG' in tail, (
        'LABEL_RESULTING_FILE does not carry the relaxed-matching tag, so a '
        'relaxed run overwrites the strict sample under its own name')
