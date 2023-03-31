import core.damages
from utils.config import Config

config = Config()

targets = core.targets.Targets()
targets.load_contracts(config.get('DIR_CONTRACTS'))
targets.load_damages(config.get('DIR_CLAIMS'))

