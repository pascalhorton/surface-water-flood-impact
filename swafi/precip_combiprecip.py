"""
Class to handle the precipitation data from CombiPrecip.
"""

from glob import glob
import xarray as xr

from .config import Config
from .precip_archive import PrecipitationArchive

config = Config()


class CombiPrecip(PrecipitationArchive):
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

    def __init__(self, year_start, year_end, cid_file=None):
        """
        The Precipitation class for CombiPrecip data. Must be netCDF files.

        Parameters
        ----------
        year_start: int
            The start year of the data
        year_end: int
            The end year of the data
        cid_file: str|None
            The path to the CID file
        """
        super().__init__(year_start, year_end, cid_file)
        self.dataset_name = "CombiPrecip"

    def prepare_data(self, data_path=None, resolution=1, time_step=1):
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
        if not self.data_path:
            raise FileNotFoundError("The data path was not provided.")
        self.resolution = resolution
        self.time_step = time_step

        files = sorted(glob(f"{self.data_path}/*.nc"))
        data = xr.open_mfdataset(
            files,
            parallel=False,
            chunks={'time': 1000}
        )
        data = data.rename_vars({'CPC': 'precip'})
        data = data.rename({'REFERENCE_TS': 'time'})

        self._generate_pickle_files(data)
