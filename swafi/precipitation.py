"""
Class to handle the precipitation data.
"""

from glob import glob
import xarray as xr

from swafi.config import Config
from swafi.domain import Domain

config = Config()


class Precipitation:
    def __init__(self, data_path, cid_file=None, dataset='CombiPrecip'):
        """
        The Precipitation class. Used for CombiPrecip data. Must be netCDF files.
        """
        if not cid_file:
            cid_file = config.get('CID_PATH')

        self.x_axis = 'x'
        self.y_axis = 'y'
        self.time_axis = 'time'
        if dataset == 'CombiPrecip':
            self.time_axis = 'REFERENCE_TS'

        self.data = None
        self.domain = Domain(cid_file)

        self._load_data(data_path)

    def get_time_series(self, cid, start, end, size=3):
        """
        Extract the precipitation time series for the given cell ID and the given
        period (between start and end).

        Parameters
        ----------
        cid: int
            The cell ID
        start: datetime.datetime
            The start of the period to extract
        end: datetime.datetime
            The end of the period to extract
        size: int
            The number of pixels to average on (default: 3x3)

        Returns
        -------
        np.array
            The timeseries as a numpy array
        """
        x, y = self.domain.get_cid_coordinates(cid)
        dx = self.domain.resolution[0]
        dy = self.domain.resolution[1]
        dpx = (size - 1) / 2

        dat = self.data.sel({self.x_axis: slice(x - dx * dpx, x + dx * dpx),
                             self.y_axis: slice(y + dy * dpx, y - dy * dpx),
                             self.time_axis: slice(start, end)})

        return dat.CPC.mean(dim=[self.x_axis, self.y_axis]).to_numpy()

    def _load_data(self, data_path):
        files = sorted(glob(f"{data_path}/*.nc"))
        self.data = xr.open_mfdataset(files, parallel=False)
