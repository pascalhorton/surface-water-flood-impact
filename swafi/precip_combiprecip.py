"""
Class to handle the precipitation data.
"""

from glob import glob
import xarray as xr

from .config import Config
from .domain import Domain
from .precip import Precipitation

config = Config()


class CombiPrecip(Precipitation):
    # Missing dates
    missing = [
        ('2004-12-31', '2005-01-01'),
        ('2005-01-16', '2005-01-16'),
        ('2005-01-21', '2005-01-21'),
        ('2009-04-29', '2009-04-29'),
        ('2016-12-06', '2016-12-06'),
        ('2017-04-07', '2017-04-07'),
        ('2017-04-12', '2017-04-12'),
        ('2017-10-06', '2017-10-07'),
        ('2021-04-07', '2021-04-07'),
        ('2022-06-27', '2022-06-30'),
        ('2022-08-16', '2022-08-21'),
        ('2022-10-17', '2022-10-23'),
    ]

    def __init__(self, data_path, cid_file=None):
        """
        The Precipitation class for CombiPrecip data. Must be netCDF files.
        """
        super().__init__(cid_file)

        self.x_axis = 'x'
        self.y_axis = 'y'
        self.time_axis = 'REFERENCE_TS'
        self.precip_var = 'CPC'

        self._load_data(data_path)

    def _load_data(self, data_path):
        files = sorted(glob(f"{data_path}/*.nc"))
        self.data = xr.open_mfdataset(files, parallel=False)
