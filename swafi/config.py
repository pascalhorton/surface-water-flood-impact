import os
from pathlib import Path
import yaml

"""
Class to handle the configuration, such as path to data and directories. It reads a 
config.yaml file containing these options.
"""


class Config:
    def __init__(self, output_dir=None):
        self.config = None
        self.output_dir = None
        filepath = ''
        if Path('config.yaml').exists():
            filepath = 'config.yaml'
        elif Path('../config.yaml').exists():
            filepath = '../config.yaml'
        elif Path('../../config.yaml').exists():
            filepath = '../../config.yaml'
        else:
            raise RuntimeError("The config.yaml file was not found.")

        with open(filepath) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            self.create_output_directory(output_dir)

    def create_output_directory(self, output_dir):
        if 'OUTPUT_DIR' in self.config:
            self.output_dir = Path(self.config['OUTPUT_DIR'])
        else:
            self.output_dir = Path(os.getcwd())

        if output_dir is not None:
            self.output_dir = self.output_dir / output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key, default=None, do_raise=True):
        if key not in self.config:
            if default:
                return default
            if not do_raise:
                return None
            raise RuntimeError(f"The entry '{key}' was not found in the config file.")
        return self.config[key]

    def set(self, key, value):
        self.config[key] = value
