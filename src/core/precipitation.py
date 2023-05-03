"""
Class to handle the precipitation data.
"""

from glob import glob
import xarray as xr

from utils.config import Config
from core.domain import Domain

config = Config()


class Precipitation:
    def __init__(self, data_path, cid_file=None):
        """
        The Precipitation class. Used for CombiPrecip data. Must be netCDF files.
        """
        if not cid_file:
            cid_file = config.get('CID_PATH')

        self.data = None
        self.domain = Domain(cid_file)

        self._load_data(data_path)

    def get_time_series(self, cid, start, end):

        pass

    def _load_data(self, data_path):
        files = sorted(glob(f"{data_path}/*.nc"))
        self.data = xr.open_mfdataset(files, parallel=True)
