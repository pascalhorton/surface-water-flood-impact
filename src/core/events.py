"""
Class to handle the events.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.config import Config

config = Config()


class Events:
    def __init__(self):
        """
        The Events class.
        """

    def load_from_parquet(self, path):
        pass