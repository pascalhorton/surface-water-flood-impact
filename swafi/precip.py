"""
Class to handle the precipitation data.
"""

from glob import glob
import xarray as xr

from .config import Config
from .domain import Domain

config = Config()


class Precipitation:
    def __init__(self, cid_file=None):
        """
        The generic Precipitation class. Must be netCDF files as relies on xarray.
        """
        if not cid_file:
            cid_file = config.get('CID_PATH')

        self.x_axis = 'x'
        self.y_axis = 'y'
        self.time_axis = 'time'
        self.precip_var = 'prec'

        self.data = None
        self.missing = None
        self.domain = Domain(cid_file)

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

        return dat[self.precip_var].mean(dim=[self.x_axis, self.y_axis]).to_numpy()

