import core.damages
import core.events
from utils.config import Config

config = Config()

damages = core.damages.Damages()
damages.load_contracts(config.get('DIR_CONTRACTS'))
damages.load_claims(config.get('DIR_CLAIMS'))
damages.select_all_categories()

events = core.events.Events()
events.load_from_parquet(config.get('EVENTS_PATH'))
