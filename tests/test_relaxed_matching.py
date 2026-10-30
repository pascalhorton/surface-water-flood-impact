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


def test_every_inference_script_keys_its_grid_on_the_detection_tag():
    """A prediction grid's name must carry the catalogue it was made against.

    The catalogue decides which cell-days become candidates at all, so the
    same model against two catalogues gives two different grids - and nothing
    inside a .nc records which catalogue produced it. Every use_* script
    short-circuits on `output_path.exists()`, so without the tag the second
    run silently returns the first one's grid. `12 Catalogue at inference/`
    fixed this in its own comparison script; the fix was never carried back
    into the inference scripts until 2026-10-29.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / 'scripts' / 'impact_functions'
    scripts = sorted(root.glob('use_*_occurrence.py'))
    assert scripts, "no inference scripts found"
    for path in scripts:
        src = path.read_text(encoding='utf-8')
        assert 'detection_tag_in' in src, (
            f"{path.name} does not derive a detection tag")
        assert 'det_tag' in src, (
            f"{path.name} does not put the detection tag in the grid name")
        # The tag must reach the file name, not merely be computed.
        assert '{det_tag}' in src, (
            f"{path.name} computes a detection tag but leaves it out of the "
            f"output path")


def test_the_random_forest_grid_name_carries_the_run_name():
    """Without it every RF wrote to one file name.

    That is why the grids in '09 Independent predictions' carry hand-added
    '_v1'/'_v2'/'_v10' suffixes: the script overwrote, so the only way to keep
    two models' predictions was to rename them outside it.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / 'scripts'
           / 'impact_functions' / 'use_rf_occurrence.py').read_text(
        encoding='utf-8')
    assert '{options.run_name}' in src, (
        "use_rf_occurrence.py must put run_name in the grid name")


def test_cnn_inference_takes_the_model_as_an_argument():
    """Ten models cannot be run from a hard-coded constant.

    use_cnn_occurrence.py drove MODEL and DATASET from module constants, so a
    batch of seeds meant editing the file between runs - which is how a server
    array job silently runs the same model ten times. --model and --dataset now
    override them, and --check-inputs resolves every path without computing.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / 'scripts'
           / 'impact_functions' / 'use_cnn_occurrence.py').read_text(
        encoding='utf-8')
    assert "'--model'" in src and "'--dataset'" in src, (
        "the model and dataset must be settable from the command line")
    assert "'--check-inputs'" in src, (
        "a pre-flight check is what keeps an array job from failing an hour "
        "in on a missing path")
    assert 'keras.models.load_model(args.model)' in src, (
        "the loaded model must come from the argument, not the constant")
    assert 'USE_TRAINING_REFERENCE' not in src, (
        "the constant was replaced by --no-training-reference; leaving it "
        "behind invites editing a value that no longer has any effect")


def _brute_aggregate(ds_pred, ds_damages, region_map, region_ids,
                     ignore_removed, relax_days):
    """Reference implementation of aggregate_by_regions: a plain loop.

    The library version groups the cells once and reduces with `reduceat`,
    because a per-region numpy call inside the day loop is 731 x 3952 small
    operations for the ZIP aggregation and does not finish. This is the
    obvious implementation it replaced, kept here so the fast one is checked
    against something written independently of it.
    """
    import numpy as np
    from swafi.utils.verification import relax_prediction_days

    n_time = len(ds_pred['time'].values)
    n_regions = len(region_ids)
    flat_region = region_map.ravel()
    region_to_idx = {rid: np.where(flat_region == rid)[0] for rid in region_ids}

    y_true = np.full((n_time, n_regions), np.nan, dtype=np.float32)
    y_pred = np.full((n_time, n_regions), np.nan, dtype=np.float32)
    has_removed = 'removed_claims' in ds_damages.variables

    for t in range(n_time):
        pflat = ds_pred['predict'].isel(time=t).to_numpy().ravel()
        cflat = ds_damages['claims'].isel(time=t).to_numpy().astype(float).ravel()
        if has_removed and ignore_removed:
            rem = ds_damages['removed_claims'].isel(time=t).to_numpy().ravel()
            cflat[rem == 1] = np.nan
        cbin = ((~np.isnan(cflat)) & (cflat > 0)).astype(np.uint8)

        for r, rid in enumerate(region_ids):
            idxs = region_to_idx[rid]
            pvals = pflat[idxs]
            y_pred[t, r] = (np.nan if np.isnan(pvals).all()
                            else np.nanmax(pvals))
            # A region-day with no insured cell is NaN, not a negative: no
            # claim is possible there, which is why the pixel path drops it.
            if np.isnan(cflat[idxs]).all():
                y_true[t, r] = np.nan
            else:
                y_true[t, r] = 1 if np.any(cbin[idxs] == 1) else 0

    if relax_days:
        for r in range(n_regions):
            relax_prediction_days(y_true[:, r], y_pred[:, r])

    return y_true, y_pred


def test_vectorised_region_aggregation_matches_the_plain_loop():
    """The reduceat version must produce the identical arrays.

    It replaced a per-region loop purely for speed, on numbers that are going
    into a published comparison table, so equivalence is not something to take
    on faith.
    """
    import numpy as np
    import xarray as xr
    from swafi.utils.aggregation import aggregate_by_regions

    rng = np.random.default_rng(7)
    n_time, ny, nx = 9, 4, 5

    pred = rng.random((n_time, ny, nx)).astype(np.float32)
    # NaNs in the predictions, including one region that is entirely NaN.
    pred[:, 0, 0] = np.nan
    pred[3, 2, :] = np.nan

    claims = np.zeros((n_time, ny, nx), dtype=np.float32)
    claims[2, 1, 1] = 1
    claims[4, 3, 4] = 2
    claims[6, 0, 2] = 1
    claims[:, 2, 2] = np.nan            # outside the insured domain

    removed = np.zeros((n_time, ny, nx), dtype=np.float32)
    removed[2, 1, 2] = 1

    def ds(**arrays):
        return xr.Dataset(
            {k: (('time', 'y', 'x'), v) for k, v in arrays.items()},
            coords={'time': np.arange(n_time)})

    # Four regions, one cell left outside every polygon (-1).
    region_map = np.array([[0, 0, 1, 1, -1],
                           [0, 0, 1, 1, 2],
                           [3, 3, 3, 2, 2],
                           [3, 3, 3, 2, 2]], dtype=np.int32)
    region_ids = [0, 1, 2, 3]

    for ignore_removed in (False, True):
        for relax_days in (False, True):
            got = aggregate_by_regions(
                ds(predict=pred.copy()),
                ds(claims=claims.copy(), removed_claims=removed.copy()),
                region_map, region_ids,
                ignore_removed=ignore_removed, relax_days=relax_days)
            want = _brute_aggregate(
                ds(predict=pred.copy()),
                ds(claims=claims.copy(), removed_claims=removed.copy()),
                region_map, region_ids, ignore_removed, relax_days)
            for name, a, b in (('y_true', got[0], want[0]),
                               ('y_pred', got[1], want[1])):
                assert np.allclose(a, b, equal_nan=True), (
                    f"{name} differs at ignore_removed={ignore_removed}, "
                    f"relax_days={relax_days}")


def test_region_ids_need_not_be_contiguous():
    """The region ids come from a shapefile index, not from range(n).

    `reduceat` works on positions, so the ids have to be mapped to positions
    first; using them as positions would index out of bounds or silently mix
    regions up.
    """
    import numpy as np
    import xarray as xr
    from swafi.utils.aggregation import aggregate_by_regions

    n_time = 3
    region_map = np.array([[700, 700, 1205],
                           [9, 9, 1205]], dtype=np.int32)
    region_ids = [9, 700, 1205]

    pred = np.zeros((n_time, 2, 3), dtype=np.float32)
    pred[1, 1, 0] = 0.9          # region 9
    pred[1, 0, 2] = 0.4          # region 1205
    claims = np.zeros((n_time, 2, 3), dtype=np.float32)

    def ds(**arrays):
        return xr.Dataset(
            {k: (('time', 'y', 'x'), v) for k, v in arrays.items()},
            coords={'time': np.arange(n_time)})

    _, y_pred = aggregate_by_regions(
        ds(predict=pred), ds(claims=claims), region_map, region_ids,
        ignore_removed=False, relax_days=False)

    # Columns follow the ORDER of region_ids: 9, 700, 1205.
    assert y_pred[1, 0] == np.float32(0.9), "region 9 should hold its own max"
    assert y_pred[1, 1] == 0.0, "region 700 has no prediction that day"
    assert y_pred[1, 2] == np.float32(0.4)


def test_a_region_with_no_insured_cell_is_not_a_negative():
    """It cannot produce a claim, so it is not evidence of a correct negative.

    NaN in `claims` means the cell carries no insurance. The pixel path drops
    those cell-days with `~np.isnan(y_true)`; the region path used to score
    them as 0. At the scales in use that was not a detail: the ZIP shapefile
    covers Switzerland while GVZ covers one canton, so 3,618 of the 3,952 ZIP
    regions overlapping the grid (92%) hold no insured cell, and counting them
    put 2.8 M samples into a comparison whose pixel twin has 1.29 M.
    """
    import numpy as np
    import xarray as xr
    from swafi.utils.aggregation import aggregate_by_regions

    n_time = 4
    # Region 0 is insured, region 1 is entirely outside the insured domain.
    region_map = np.array([[0, 0, 1, 1]], dtype=np.int32)
    region_ids = [0, 1]

    claims = np.zeros((n_time, 1, 4), dtype=np.float32)
    claims[:, 0, 2:] = np.nan          # region 1: never insured
    claims[2, 0, 0] = 1                # region 0: a claim on day 2

    pred = np.full((n_time, 1, 4), 0.3, dtype=np.float32)

    def ds(**arrays):
        return xr.Dataset(
            {k: (('time', 'y', 'x'), v) for k, v in arrays.items()},
            coords={'time': np.arange(n_time)})

    y_true, y_pred = aggregate_by_regions(
        ds(predict=pred), ds(claims=claims), region_map, region_ids,
        ignore_removed=False, relax_days=False)

    assert np.all(np.isnan(y_true[:, 1])), (
        "an uninsured region must be NaN on every day, not a correct negative")
    assert y_true[2, 0] == 1
    assert np.array_equal(y_true[[0, 1, 3], 0], [0, 0, 0]), (
        "the insured region keeps its real negatives")

    # And the prediction is still there, so the NaN comes from the truth side
    # only - a model is not credited or penalised for the region either way.
    assert not np.isnan(y_pred[0, 1])


def test_grid_scoring_does_not_need_the_pickled_domain():
    """Comparing two local netCDFs must not depend on a network share.

    `Damages.from_nc_file` is `xr.open_dataset` plus a few attributes set on
    an object the scorer never reads, but constructing DamagesGvz first loads
    the pickled domain out of PICKLES_DIR. That made every row of a 136-row
    run fail with
        FileNotFoundError: <PICKLES_DIR>/domain.pickle
    when the share was unreachable - which it is from any machine off the
    institute network. Only 'claims' and 'removed_claims' are used and both
    are in the grid file.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / 'scripts'
           / 'impact_functions' / 'score_prediction_grids.py').read_text(
        encoding='utf-8')
    # Usage, not mention: the comment in the script explains why those names
    # are absent, so a plain substring test matches its own documentation.
    for forbidden in ('swafi.damages', '.from_nc_file('):
        assert forbidden not in src, (
            f"score_prediction_grids.py must not use {forbidden}: it drags in "
            f"PICKLES_DIR for a file it then ignores")
    assert 'xr.open_dataset(path)' in src


def test_a_run_that_scored_nothing_exits_non_zero():
    """136 error rows used to exit 0.

    The CSV was written, the summary printed, and the shell saw success - so a
    scheduled or backgrounded run would have been read as having produced
    numbers.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / 'scripts'
           / 'impact_functions' / 'score_prediction_grids.py').read_text(
        encoding='utf-8')
    assert "df['error'].notna().all()" in src, (
        "a run where every row failed must not report success")
    assert 'raise SystemExit(1)' in src
