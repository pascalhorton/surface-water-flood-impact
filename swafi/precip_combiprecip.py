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

    def __init__(self, cid_file=None, data_path=None):
        """
        The Precipitation class for CombiPrecip data. Must be netCDF files.

        Parameters
        ----------
        cid_file: str|None
            The path to the CID file
        data_path: str|None
            The path to the data files
        """
        super().__init__(cid_file, data_path)

    def load_data(self, data_path=None, resolution=1, time_step=1):
        """
        Load the precipitation data from the given path.

        Parameters
        ----------
        data_path: str|None
            The path to the data files
        resolution: int
            The resolution [km] of the precipitation data (default: 1)
        time_step: int
            The time step [h] of the precipitation data (default: 1)
        """
        if data_path:
            self.data_path = data_path
        if not self.data_path:
            self.data_path = config.get('DIR_PRECIP')
        self.resolution = resolution
        self.time_step = time_step

        files = sorted(glob(f"{data_path}/*.nc"))
        self.data = xr.open_mfdataset(files, parallel=False, chunks={'time': 1000})
        self.data = self.data.rename_vars({'CPC': 'precip'})
        self.data = self.data.rename({'REFERENCE_TS': 'time'})

        self._load_from_pickle()

        assert len(self.data.dims) == 3, "Precipitation must be 3D"
