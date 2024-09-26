"""
Class to handle the precipitation data from ICON.
"""

from glob import glob
import xarray as xr

from .config import Config
from .precip_forecast import PrecipitationForecast

config = Config()


class IconPrecip(PrecipitationForecast):

    def __init__(self, cid_file=None):
        """
        The Precipitation class for ICON data.

        Parameters
        ----------
        cid_file: str|None
            The path to the CID file
        """
        super().__init__(cid_file)
        self.dataset_name = "ICON"

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
        if not self.data_path:
            raise FileNotFoundError("The data path was not provided.")
        self.resolution = resolution
        self.time_step = time_step

        files = sorted(glob(f"{self.data_path}/*.nc"))
        self.data = xr.open_mfdataset(files, parallel=False, chunks={'time': 1000})
        self.data.rename_vars({'CPC': 'precip'}, inplace=True)
        self.data.rename({'REFERENCE_TS': 'time'}, inplace=True)

        raise NotImplementedError("This method must be implemented.")
