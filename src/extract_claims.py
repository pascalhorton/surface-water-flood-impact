import core.damages
from utils.config import Config

config = Config()

damages = core.damages.Damages()
damages.load_contracts(config.get('DIR_CONTRACTS'))
damages.load_claims(config.get('DIR_CLAIMS'))

