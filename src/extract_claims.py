import core.damages
import core.events
from utils.config import Config

config = Config()

damages = core.damages.Damages(cid_file=config.get('CID_PATH'),
                               dir_contracts=config.get('DIR_CONTRACTS'),
                               dir_claims=config.get('DIR_CLAIMS'))
damages.select_all_categories()

events = core.events.Events()
events.load_events_and_select_locations(config.get('EVENTS_PATH'), damages)

damages.match_with_events(events)