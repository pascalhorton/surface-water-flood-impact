from swafi.impact import Impact
from swafi.impact_basic_options import ImpactBasicOptions


def _make_options():
    options = ImpactBasicOptions()
    options.event_method = 'simple'
    options.target_type = 'occurrence'
    options.random_state = 42
    options.use_event_attributes = True
    options.use_static_attributes = False
    return options


def test_update_potential_features_adds_subhourly_features():
    impact = Impact(_make_options())  # no events, like the use_* scripts

    # Without events, the defaults contain no sub-hourly features
    assert 'p_10min_q' not in impact.tabular_features['event']

    # 5-min events: the sub-hourly features must appear
    impact.update_potential_features(
        ['e_date', 'i_max_q', 'p_5min_q', 'p_10min_q', 'p_20min_q',
         'p_30min_q', 'p_1h_q'])
    for feature in ('p_10min_q', 'p_20min_q', 'p_30min_q', 'p_1h_q'):
        assert feature in impact.tabular_features['event']

    # Hourly events: back to the defaults without sub-hourly features
    impact.update_potential_features(['e_date', 'i_max_q'])
    assert 'p_10min_q' not in impact.tabular_features['event']
