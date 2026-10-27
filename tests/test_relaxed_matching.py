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


# ---------------------------------------------------------------------------
# Evaluating the two legs against ONE target set
# ---------------------------------------------------------------------------

def _split_df(match_steps, targets=None):
    n = len(match_steps)
    if targets is None:
        targets = [1 if s else 0 for s in match_steps]
    return pd.DataFrame({
        'target': [float(t) for t in targets],
        'nb_claims': [1.0 if s else 0.0 for s in match_steps],
        'match_step': match_steps,
    })


class _Opts:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_eval_on_strict_positives_demotes_only_the_relaxed_ones():
    from swafi.impact import Impact
    imp = Impact.__new__(Impact)
    imp.options = _Opts(eval_on_strict_positives=True)
    out = imp._demote_relaxed_positives(_split_df([1, 2, 0, 2]), 'valid')
    assert out['target'].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert out['nb_claims'].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_eval_on_strict_positives_does_not_mutate_the_caller():
    """val_df is a slice of self.df; writing through it would demote the
    TRAINING labels too and silently turn this leg into the other one."""
    from swafi.impact import Impact
    imp = Impact.__new__(Impact)
    imp.options = _Opts(eval_on_strict_positives=True)
    df = _split_df([1, 2])
    imp._demote_relaxed_positives(df, 'valid')
    assert df['target'].tolist() == [1.0, 1.0], 'the input was modified in place'


def test_eval_on_strict_positives_needs_the_column():
    from swafi.impact import Impact
    imp = Impact.__new__(Impact)
    imp.options = _Opts(eval_on_strict_positives=True)
    df = _split_df([1, 2]).drop(columns='match_step')
    with pytest.raises(ValueError, match='match_step'):
        imp._demote_relaxed_positives(df, 'valid')


def test_feature_cache_key_separates_the_two_legs():
    """The two legs differ ONLY in which events are positive: same shape, same
    columns, same first row. Without the target in the key the second run
    reloads the first one's labels and the comparison is a model against
    itself. use_pickle is False everywhere today, so this is a latent trap -
    pinned so it stays shut."""
    from pathlib import Path

    from swafi.impact import Impact

    def key_for(targets):
        imp = Impact.__new__(Impact)
        imp.df = pd.DataFrame({
            'cid': [1, 2, 3],
            'target': targets,
            'nb_claims': [0.0, 0.0, 0.0],
            'match_step': [0, 1, 2],
        })
        imp.features = ['cid']
        imp.tmp_dir = Path('.')
        return imp._create_data_tmp_file_name(['f.csv']).name

    a = key_for([0.0, 1.0, 1.0])   # relaxed positive kept
    b = key_for([0.0, 1.0, 0.0])   # relaxed positive demoted
    assert a != b, 'the feature cache cannot tell the two legs apart'


def test_cnn_full_domain_scoring_matches_the_other_models():
    """use_cnn_occurrence.py scored strict while use_lr/rf/ann/thr scored
    relaxed, so CNN full-domain numbers were measured under a harsher rule
    than the baselines'. Any table putting them side by side understated the
    CNN. Pinned so it cannot drift apart again unnoticed."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / 'scripts' / 'impact_functions'
    src = (root / 'use_cnn_occurrence.py').read_text(encoding='utf-8')
    assert 'RELAX_DAYS = True' in src
    assert 'IGNORE_REMOVED = True' in src
    assert 'relax_days=False' not in src, (
        'a hard-coded relax_days=False is back in the CNN inference script')


# ---------------------------------------------------------------------------
# Which neighbour donates the prediction, when the claim date is relaxed
# ---------------------------------------------------------------------------

def _grid(n_days, pred, true):
    """One cell, n_days long: (1, 1) spatial shape as the assessment expects."""
    import numpy as np
    import xarray as xr

    def ds(name, values):
        arr = np.array(values, dtype=float).reshape(n_days, 1, 1)
        return xr.Dataset({name: (('time', 'y', 'x'), arr)})

    return ds('predict', pred), ds('claims', true)


def test_relaxation_prefers_the_day_before():
    """With both neighbours available, the BACKWARD one donates: 58% of
    mis-dated GVZ claims have their storm before the recorded date, and a storm
    after a claim cannot have caused it."""
    from swafi.utils.verification import prepare_full_domain_assessment

    #        day:   0    1    2
    # claim on day 1; a prediction on each side, neither carrying a claim
    ds_pred, ds_dam = _grid(3, [0.7, 0.0, 0.9], [0, 1, 0])
    y_true, y_pred = prepare_full_domain_assessment(
        ds_pred, ds_dam, ignore_removed=False, relax_days=True, flatten=False)

    assert y_pred[1, 0, 0] == 0.7, 'the day-after donated instead of the day-before'
    assert y_pred[0, 0, 0] == 0.0, 'the donor day was not cleared'
    assert y_pred[2, 0, 0] == 0.9, 'the day-after was consumed as well'


def test_relaxation_still_uses_the_day_after_when_alone():
    """The preference is an ordering, not an exclusion: a claim dated one day
    early still gets its prediction."""
    from swafi.utils.verification import prepare_full_domain_assessment

    ds_pred, ds_dam = _grid(3, [0.0, 0.0, 0.9], [0, 1, 0])
    _y_true, y_pred = prepare_full_domain_assessment(
        ds_pred, ds_dam, ignore_removed=False, relax_days=True, flatten=False)

    assert y_pred[1, 0, 0] == 0.9
    assert y_pred[2, 0, 0] == 0.0


def test_relaxation_leaves_a_neighbour_that_carries_its_own_claim():
    """A neighbour with a claim of its own needs its prediction: taking it
    would turn one hit into another and invent a miss."""
    from swafi.utils.verification import prepare_full_domain_assessment

    ds_pred, ds_dam = _grid(3, [0.7, 0.0, 0.0], [1, 1, 0])
    _y_true, y_pred = prepare_full_domain_assessment(
        ds_pred, ds_dam, ignore_removed=False, relax_days=True, flatten=False)

    assert y_pred[0, 0, 0] == 0.7, 'a neighbour with its own claim was robbed'
    assert y_pred[1, 0, 0] == 0.0


def test_relaxation_does_not_change_the_ceiling_by_preference():
    """The donor preference moves false alarms, not the ceiling: the claim day
    gets a prediction whichever side donates it."""
    from swafi.utils.verification import prepare_full_domain_assessment

    for pred in ([0.7, 0.0, 0.9], [0.9, 0.0, 0.7]):
        ds_pred, ds_dam = _grid(3, pred, [0, 1, 0])
        y_true, y_pred = prepare_full_domain_assessment(
            ds_pred, ds_dam, ignore_removed=False, relax_days=True,
            flatten=True)
        unreachable = int(((y_true > 0) & (y_pred <= 0)).sum())
        assert unreachable == 0


def test_a_removed_cell_day_cannot_donate_its_prediction():
    """ignore_removed lowers the recall CEILING, not only the sample.

    It NaNs the cell-days whose claim the category selection dropped, and a day
    whose y_true is NaN fails `y_true[neighbour] == 0`, so it stops being an
    eligible donor. Measured on the gvz 2023-2024 grid: 329 positives either
    way, 55 unreachable at ignore_removed=False against 57 at True. That is
    right - a removed cell-day saw water damage of another category, so it is
    not a clean negative - but it means a ceiling is ambiguous unless BOTH
    flags are quoted with it.
    """
    import numpy as np
    import xarray as xr
    from swafi.utils.verification import prepare_full_domain_assessment

    #         day:    0    1
    # The claim is on day 1 and carries no prediction. Day 0 holds the only
    # prediction that could be moved onto it, and day 0 is a removed cell-day.
    def ds(pairs):
        out = {}
        for name, values in pairs.items():
            out[name] = (('time', 'y', 'x'),
                         np.array(values, dtype=float).reshape(2, 1, 1))
        return xr.Dataset(out)

    ds_pred = ds({'predict': [0.8, 0.0]})
    ds_dam = ds({'claims': [0, 1], 'removed_claims': [1, 0]})

    _, y_pred = prepare_full_domain_assessment(
        ds_pred, ds_dam, ignore_removed=True, relax_days=True, flatten=False)
    assert y_pred[1, 0, 0] == 0, (
        "a removed cell-day must not donate its prediction")

    # With the same grid and ignore_removed off, day 0 is a clean negative and
    # the move goes through - so the difference is the flag, not the geometry.
    ds_pred = ds({'predict': [0.8, 0.0]})
    ds_dam = ds({'claims': [0, 1], 'removed_claims': [1, 0]})
    _, y_pred = prepare_full_domain_assessment(
        ds_pred, ds_dam, ignore_removed=False, relax_days=True, flatten=False)
    assert y_pred[1, 0, 0] == 0.8


def test_the_region_path_uses_the_same_relaxation_as_the_pixel_path():
    """They used to hold two copies that preferred opposite neighbours.

    `assess_aggregation.py` carried its own relaxation loop, whose comment
    claimed to "emulate pixel-level logic" while trying the day AFTER first -
    so when the pixel-level preference was flipped to the day before, the
    region-level numbers in `Scores gvz.xlsx` silently stopped matching. Both
    now call `relax_prediction_days`, and this pins that there is one
    implementation rather than two that happen to agree today.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / 'swafi' / 'utils'
           / 'aggregation.py').read_text(encoding='utf-8')
    assert 'relax_prediction_days' in src, (
        "the region path must call the shared relaxation helper")
    assert 'end_day - 1' not in src and 'start_day' not in src, (
        "the region path must not carry its own copy of the donor logic")

    agg = (Path(__file__).resolve().parents[1] / 'scripts'
           / 'impact_functions' / 'assess_aggregation.py').read_text(
        encoding='utf-8')
    assert '_aggregate_by_regions' not in agg, (
        "assess_aggregation must use the library helper, not a local copy")
    assert 'from swafi.utils.aggregation import' in agg


def test_relax_days_and_ignore_removed_are_separate_settings():
    """One constant used to drive both in assess_aggregation.py.

    That is why the score table's block labels only ever read both-False or
    both-True, although the flags answer different questions.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / 'scripts'
           / 'impact_functions' / 'assess_aggregation.py').read_text(
        encoding='utf-8')
    assert 'RELAX_DAYS' in src and 'IGNORE_REMOVED' in src
    assert 'ignore_removed=RELAX,' not in src, (
        "the two flags must not be driven by one constant")


def test_scoring_reopens_the_datasets_between_passes():
    """prepare_full_domain_assessment moves predictions IN PLACE.

    `.to_numpy()` on a loaded DataArray hands back the array itself, so a
    caller that scores one open dataset under several settings scores the later
    ones against already-moved predictions. Measured on a real grid: the sum of
    `predict` changes after one pass. Both scoring scripts must therefore open
    the datasets inside their own loop.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / 'scripts' / 'impact_functions'
    for name in ('assess_aggregation.py', 'score_prediction_grids.py'):
        src = (root / name).read_text(encoding='utf-8')
        assert 'xr.open_dataset' in src, name
        # The open must not sit before the loop that varies the setting.
        before_loop = src.split('for aggregation in')[0]
        assert 'ds_pred = xr.open_dataset(result_path)' not in before_loop, (
            f"{name} opens the prediction grid outside its aggregation loop, "
            f"so later passes read predictions an earlier pass moved")


def test_the_in_place_hazard_is_recorded_where_it_bites():
    """A future reader must not have to rediscover it."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / 'swafi' / 'utils'
           / 'verification.py').read_text(encoding='utf-8')
    assert 'IN PLACE' in src, (
        "prepare_full_domain_assessment must document that it mutates ds_pred")
