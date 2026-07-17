import pickle

from swafi.impact_basic_options import ImpactBasicOptions
from swafi.impact_rf import ImpactRandomForest


def _make_options():
    options = ImpactBasicOptions()
    options.event_method = 'simple'
    options.target_type = 'occurrence'
    options.random_state = 42
    options.use_event_attributes = True
    options.use_static_attributes = False
    options.run_name = 'test'
    return options


def test_rf_save_load_roundtrip_keeps_features(tmp_path):
    rf = ImpactRandomForest(_make_options())
    rf.model = 'fake-model'  # save/load only pickles it, any object works
    rf.features = ['event:i_max_q', 'event:p_10min_q']
    rf.save_model(str(tmp_path), 'model_rf')

    rf_loaded = ImpactRandomForest(_make_options())
    rf_loaded.load_model(str(tmp_path), 'model_rf')
    assert rf_loaded.model == 'fake-model'
    assert rf_loaded.features == ['event:i_max_q', 'event:p_10min_q']


def test_rf_load_legacy_bare_model(tmp_path):
    with open(tmp_path / 'model_rf_test.pkl', 'wb') as f:
        pickle.dump('bare-model', f)

    rf = ImpactRandomForest(_make_options())
    rf.load_model(str(tmp_path), 'model_rf')
    assert rf.model == 'bare-model'
    assert rf.features == []
