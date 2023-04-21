import core.damages
import core.events
from utils.config import Config

config = Config()

criteria_to_test = [
    ['i_mean'],
    ['i_max'],
    ['p_sum'],
    ['r_ts_win'],
    ['r_ts_evt'],
    ['i_mean', 'i_max', 'p_sum'],
    ['i_mean', 'i_max', 'p_sum', 'r_ts_win'],
    ['i_mean', 'i_max', 'p_sum', 'r_ts_evt'],
    ['i_mean', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'],
]

for i, criteria in enumerate(criteria_to_test):
    print(f"Assessing criteria {criteria}")
    damages = core.damages.Damages(cid_file=config.get('CID_PATH'),
                                   dir_contracts=config.get('DIR_CONTRACTS'),
                                   dir_claims=config.get('DIR_CLAIMS'))
    damages.select_all_categories()

    events = core.events.Events()
    events.load_events_and_select_locations(config.get('EVENTS_PATH'), damages)

    damages.match_with_events(events, criteria=criteria,
                              filename=f'damages_matched_conf_{i}.pickle')

