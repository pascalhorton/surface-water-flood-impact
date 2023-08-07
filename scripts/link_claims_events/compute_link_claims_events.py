from pathlib import Path
from swafi.config import Config
from swafi.damages import Damages
from swafi.events import Events

config = Config()

# Main options
label = 'original_w_prior'
criteria = ['prior', 'i_mean', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt']
window_days = [5, 3, 1]
damage_categories = ['external', 'pluvial']


filename = f'damages_linked_{label}.pickle'
file_path = Path(config.get('PICKLES_DIR') + '/' + filename)

damages = Damages(cid_file=config.get('CID_PATH'),
                  dir_contracts=config.get('DIR_CONTRACTS'),
                  dir_claims=config.get('DIR_CLAIMS'))
damages.select_categories_type(damage_categories)

events = Events()
events.load_events_and_select_locations_with_contracts(
    config.get('EVENTS_PATH'), damages)

damages.link_with_events(events, criteria=criteria, filename=filename,
                         window_days=window_days)

print("Done.")
