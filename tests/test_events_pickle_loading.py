import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import swafi.events as events_module


def _mock_config_get(tmp_path):
    def _get(key):
        if key == 'PICKLES_DIR':
            return str(tmp_path)
        raise KeyError(key)

    return _get


def _sample_events_df():
    return pd.DataFrame(
        {
            'eid': [1, 2],
            'nb_contracts': [3, 1],
            'target': [0, 1],
        }
    )


def test_load_events_uses_fallback_when_main_pickle_is_incompatible(tmp_path, monkeypatch):
    filename = 'events_test.pickle'
    df = _sample_events_df()

    monkeypatch.setattr(events_module.config, 'get', _mock_config_get(tmp_path))
    events_only_path = events_module._events_only_pickle_path(tmp_path, filename)
    df.to_pickle(events_only_path, compression='bz2')

    def _raise_incompatible(_):
        raise NotImplementedError('simulated cross-version incompatibility')

    monkeypatch.setattr(events_module, '_load_events_from_file', _raise_incompatible)

    loaded = events_module.load_events_from_pickle(filename=filename)
    assert_frame_equal(loaded.events.reset_index(drop=True), df.reset_index(drop=True))


def test_load_events_raises_clear_error_when_no_fallback_exists(tmp_path, monkeypatch):
    filename = 'events_missing_fallback.pickle'
    file_path = tmp_path / filename
    file_path.write_bytes(b'not-a-valid-pickle')

    monkeypatch.setattr(events_module.config, 'get', _mock_config_get(tmp_path))

    def _raise_incompatible(_):
        raise NotImplementedError('simulated cross-version incompatibility')

    monkeypatch.setattr(events_module, '_load_events_from_file', _raise_incompatible)

    with pytest.raises(Exception, match='Fallback file'):
        events_module.load_events_from_pickle(filename=filename)


def test_dump_always_writes_portable_events_only_pickle(tmp_path, monkeypatch):
    filename = 'events_dump_test.pickle'
    df = _sample_events_df()

    monkeypatch.setattr(events_module.config, 'get', _mock_config_get(tmp_path))

    events = events_module.Events(use_dump=False)
    events.use_dump = True
    events.events = df
    events._dump_object(filename=filename)

    events_only_path = events_module._events_only_pickle_path(tmp_path, filename)
    assert events_only_path.is_file()
    loaded_df = pd.read_pickle(events_only_path, compression='bz2')
    assert_frame_equal(loaded_df.reset_index(drop=True), df.reset_index(drop=True))



def _catalogue(n_events, first_day):
    """A minimal events frame, distinguishable by its length."""
    days = pd.date_range(first_day, periods=n_events, freq='D')
    return pd.DataFrame({
        'cid': 1,
        'e_date': days,
        'i_max_date': days + pd.Timedelta(hours=14),
        'nb_contracts': 3,
    })


def test_the_linkage_dump_is_keyed_on_the_event_definition(tmp_path, monkeypatch):
    """Re-extracting a catalogue must not be ignored by the linkage.

    load_events_and_select_those_with_contracts caches the parquet it read in
    'events_<tag>.pickle', and the tag carries the damage dataset, the method
    and the precipitation dataset. It did NOT carry how the events were
    detected, which is recorded only in the parquet's name. So pinning the
    detection step, re-extracting, and re-running the linkage reloaded the
    PREVIOUS events from that dump and never opened the new parquet - then
    wrote them out under the new label. Nothing downstream could tell: the
    file date was new, the name said the new definition, and the contents were
    the old catalogue.
    """
    monkeypatch.setattr(events_module.config, 'get', _mock_config_get(tmp_path))
    monkeypatch.setattr(events_module.Events, 'select_years_with_contracts',
                        lambda self, damages: None)
    monkeypatch.setattr(events_module.Events, 'select_locations_with_contracts',
                        lambda self, damages: None)

    before = tmp_path / ('events_cpc_5min_model_domain_2005_2024_simple'
                         '_det12h_thr10mm_centered_peakdays.parquet')
    pinned = tmp_path / ('events_cpc_5min_model_domain_2005_2024_simple'
                         '_det12h_thr10mm_centered_peakdays_step1h.parquet')
    _catalogue(500, '2010-01-01').to_parquet(before)
    _catalogue(430, '2010-01-01').to_parquet(pinned)

    tag = 'gvz_simple_5min'  # what compute_claims_events_link.py builds

    first = events_module.Events()
    first.load_events_and_select_those_with_contracts(str(before), None, tag)
    assert len(first.events) == 500

    second = events_module.Events()
    second.load_events_and_select_those_with_contracts(str(pinned), None, tag)
    assert len(second.events) == 430, (
        "the linkage reloaded the previous catalogue from its dump instead of "
        "reading the re-extracted parquet")

    # The two definitions keep separate dumps, so neither re-read invalidates
    # the other and both stay resumable.
    assert (tmp_path / f'events_{tag}_det12h_thr10mm_centered_peakdays'
                       f'.pickle').is_file()
    assert (tmp_path / f'events_{tag}_det12h_thr10mm_centered_peakdays_step1h'
                       f'.pickle').is_file()


def test_an_untagged_catalogue_keeps_its_historical_dump_name(tmp_path,
                                                              monkeypatch):
    """The classic method and the extractions predating the detection tag:
    their dumps must still be found, not re-read under a new name."""
    monkeypatch.setattr(events_module.config, 'get', _mock_config_get(tmp_path))
    monkeypatch.setattr(events_module.Events, 'select_years_with_contracts',
                        lambda self, damages: None)
    monkeypatch.setattr(events_module.Events, 'select_locations_with_contracts',
                        lambda self, damages: None)

    parquet = tmp_path / 'events_cpc_model_domain_2005_2024_classic.parquet'
    _catalogue(300, '2010-01-01').to_parquet(parquet)

    events = events_module.Events()
    events.load_events_and_select_those_with_contracts(str(parquet), None,
                                                       'gvz_classic')

    assert (tmp_path / 'events_gvz_classic.pickle').is_file()
