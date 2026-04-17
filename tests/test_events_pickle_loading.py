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
    df.to_pickle(events_only_path, compression='gzip')

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
    loaded_df = pd.read_pickle(events_only_path, compression='gzip')
    assert_frame_equal(loaded_df.reset_index(drop=True), df.reset_index(drop=True))

