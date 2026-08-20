import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(script_name=None, console_level=logging.INFO, file_level=logging.DEBUG):
    """Configure root logger: INFO to stdout, DEBUG to a dated file in OUTPUT_DIR/logs/."""
    from swafi.config import Config
    log_dir = Path(Config().get('OUTPUT_DIR')) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{script_name}_" if script_name else ""
    log_file = log_dir / f"{prefix}{timestamp}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter(fmt))
    root.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)

    for noisy in ('tensorflow', 'keras', 'optuna', 'matplotlib', 'rasterio'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging to %s", log_file)
