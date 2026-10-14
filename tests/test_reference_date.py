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
import numpy as np
import pandas as pd
import pytest

from swafi.impact import Impact


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


@pytest.mark.parametrize("ref_date", ['middle', 'end', 'i_max', 'i_max_only'])
def test_e_date_wins_whatever_ref_date_says(ref_date):
    """The event is already dated on its peak's day; ref_date is for the
    claim-based definitions, which have no e_date to use."""
    out = Impact.set_reference_date(_simple_events(), ref_date)

    assert list(out['date']) == list(
        pd.to_datetime(['2021-06-15', '2021-06-16', '2021-06-17']))


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
    import inspect

    from swafi.impact_cnn import ImpactCnn
    from swafi.impact_lstm import ImpactLstm

    for cls in (ImpactCnn, ImpactLstm):
        src = inspect.getsource(cls.get_data_generator_inference)
        assert 'set_reference_date' in src, (
            f"{cls.__name__} must take the anchor from the shared rule")
        assert "'i_max_date'" not in src, (
            f"{cls.__name__} derives its own anchor again")

    # And the shared rule is the one place that names the column.
    assert "'i_max_date'" in inspect.getsource(Impact.set_reference_date)

    # Under the simple definition it yields midnight, not the peak instant.
    out = Impact.set_reference_date(_simple_events())
    assert (out['date'].dt.hour == 0).all()
    assert not np.array_equal(
        out['date'].to_numpy(),
        pd.to_datetime(_simple_events()['i_max_date']).to_numpy())
