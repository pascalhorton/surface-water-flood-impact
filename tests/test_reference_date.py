"""
The precipitation window handed to the data generators is built around the
'date' column (see ImpactCnnDataGenerator.get_precip_window_bounds), so
training and inference have to derive it the same way. They did not: both
inference generators renamed i_max_date to 'date' themselves, while
split_sample takes e_date under the simple event definition. A model trained
on [e_date, e_date + 24 h] was therefore applied to
[i_max_date, i_max_date + 24 h] - the same length, shifted by however long
after midnight the storm peaked, which is 0 to 24 h and differs per event.

Impact.set_reference_date is now the single rule. These tests pin it, and pin
that the two callers agree.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from swafi.impact import Impact
from swafi.impact_cnn import ImpactCnn
from swafi.impact_lstm import ImpactLstm


def _simple_events():
    """The simple event definition: dated on the day of the intensity peak."""
    return pd.DataFrame({
        'cid': [1, 2, 3],
        'e_date': pd.to_datetime(['2021-06-15', '2021-06-16', '2021-06-17']),
        'i_max_date': pd.to_datetime(['2021-06-15 17:00',
                                      '2021-06-16 03:00',
                                      '2021-06-17 23:00']),
        'e_start': pd.to_datetime(['2021-06-15', '2021-06-16', '2021-06-17']),
        'e_end': pd.to_datetime(['2021-06-16', '2021-06-17', '2021-06-18']),
    })


def _claim_events():
    """A claim-based definition: no e_date, so ref_date decides."""
    return pd.DataFrame({
        'cid': [1, 2],
        'date_claim': pd.to_datetime(['2021-06-15', None]),
        'i_max_date': pd.to_datetime(['2021-06-15 17:00', '2021-06-16 03:00']),
        'e_start': pd.to_datetime(['2021-06-15', '2021-06-16']),
        'e_end': pd.to_datetime(['2021-06-16', '2021-06-17']),
    })


def test_the_simple_definition_anchors_on_midnight():
    """e_date wins, so the window 0 h to 24 h is the event day."""
    out = Impact.set_reference_date(_simple_events())

    assert 'e_date' not in out.columns
    assert list(out['date']) == list(
        pd.to_datetime(['2021-06-15', '2021-06-16', '2021-06-17']))


@pytest.mark.parametrize("ref_date", ['auto', 'e_date'])
def test_the_day_anchor_can_be_asked_for_or_left_to_default(ref_date):
    out = Impact.set_reference_date(_simple_events(), ref_date)

    assert list(out['date']) == list(
        pd.to_datetime(['2021-06-15', '2021-06-16', '2021-06-17']))


def test_the_peak_anchor_overrides_e_date_when_asked_for():
    """The capability that did not exist: comparing the two anchors.

    e_date used to win whatever ref_date said, so the peak anchor was
    unreachable under the simple definition and no experiment ever measured
    it against the day anchor.
    """
    out = Impact.set_reference_date(_simple_events(), 'i_max_only')

    assert list(out['date']) == list(pd.to_datetime(
        ['2021-06-15 17:00', '2021-06-16 03:00', '2021-06-17 23:00']))
    # e_date goes, so nothing downstream can mistake it for the anchor.
    assert 'e_date' not in out.columns


@pytest.mark.parametrize("ref_date", ['middle', 'end', 'i_max'])
def test_the_claim_modes_are_refused_without_claim_dates(ref_date):
    """They used to fall through to e_date silently, which hid the mistake."""
    with pytest.raises(ValueError, match="date_claim"):
        Impact.set_reference_date(_simple_events(), ref_date)


def test_the_day_anchor_is_refused_without_an_event_date():
    with pytest.raises(ValueError, match="e_date"):
        Impact.set_reference_date(_claim_events(), 'e_date')


def test_auto_falls_back_to_the_peak_without_an_event_date():
    out = Impact.set_reference_date(_claim_events(), 'auto')

    assert out['date'].iloc[1] == pd.Timestamp('2021-06-16 03:00')


def test_the_peak_lies_inside_the_window_it_anchors():
    """Why midnight is a sound anchor for a 24-hour window.

    The simple definition dates an event by the day its peak falls in, so the
    peak is always between the bounds - the window holds it and the rain that
    led up to it, which is what the model is being asked about.
    """
    out = Impact.set_reference_date(_simple_events())
    peaks = pd.to_datetime(_simple_events()['i_max_date'])

    offsets = (peaks - out['date']) / pd.Timedelta(hours=1)
    assert ((offsets >= 0) & (offsets <= 24)).all()


def test_a_claim_definition_still_uses_the_claim_date_at_18h():
    """Unchanged behaviour where there is no e_date to prefer."""
    out = Impact.set_reference_date(_claim_events(), 'i_max')

    assert out['date'].iloc[0] == pd.Timestamp('2021-06-15 18:00')
    # The event with no claim falls back to the peak's day, also at 18:00.
    assert out['date'].iloc[1] == pd.Timestamp('2021-06-16 18:00')


def test_i_max_only_keeps_the_peak_instant():
    out = Impact.set_reference_date(_claim_events(), 'i_max_only')

    assert out['date'].iloc[1] == pd.Timestamp('2021-06-16 03:00')


def test_an_unknown_ref_date_is_refused():
    with pytest.raises(ValueError, match="Unknown reference date"):
        Impact.set_reference_date(_claim_events(), 'whenever')


def test_no_caller_derives_the_anchor_for_itself():
    """The regression this file exists for.

    Comparing set_reference_date against itself would prove nothing, and
    building a real inference generator needs exposure, features and a
    precipitation store. What can be pinned cheaply is the cause: the rule is
    stated once, and no caller renames i_max_date behind its back. Both
    inference generators used to, which shifted every window against training
    by however long after midnight the storm peaked.
    """
    for cls in (ImpactCnn, ImpactLstm):
        src = inspect.getsource(cls.get_data_generator_inference)
        assert 'set_reference_date' in src, (
            f"{cls.__name__} must take the anchor from the shared rule")
        assert "'i_max_date'" not in src, (
            f"{cls.__name__} derives its own anchor again")
        # Calling the rule is not enough: it has a default, and taking the
        # default applies a model trained with --event-ref-date i_max_only to
        # day-anchored windows. The option has to be read.
        assert 'get_reference_date_mode()' in src, (
            f"{cls.__name__} calls set_reference_date without the anchor the "
            f"model was trained with")

    # And the shared rule is the one place that names the column.
    assert "'i_max_date'" in inspect.getsource(Impact.set_reference_date)

    # Under the simple definition it yields midnight, not the peak instant.
    out = Impact.set_reference_date(_simple_events())
    assert (out['date'].dt.hour == 0).all()
    assert not np.array_equal(
        out['date'].to_numpy(),
        pd.to_datetime(_simple_events()['i_max_date']).to_numpy())


@pytest.mark.parametrize("trained_with, expected", [
    ('auto', 'auto'),
    ('i_max_only', 'i_max_only'),
    ('e_date', 'e_date'),
])
def test_inference_reads_the_anchor_the_model_was_trained_with(trained_with,
                                                               expected):
    class Options:
        event_ref_date = trained_with

    class Model(Impact):
        def __init__(self):
            self.options = Options()

    assert Model().get_reference_date_mode() == expected


@pytest.mark.parametrize("options", [
    type('NoAttribute', (), {})(),      # saved before the option existed
    type('Unset', (), {'event_ref_date': None})(),
])
def test_a_model_without_the_option_gets_what_it_was_trained_with(options):
    """'auto' is what every model saved before the option existed used."""
    class Model(Impact):
        def __init__(self):
            self.options = options

    assert Model().get_reference_date_mode() == 'auto'
