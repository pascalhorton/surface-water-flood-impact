"""
Class to handle the precipitation forecast data.
"""
from pathlib import Path

from .config import Config
from .domain import Domain

config = Config()


class Precipitation:
    def __init__(self, cid_file=None):
        """
        The generic PrecipitationForecast class.

        Parameters
        ----------
        cid_file: str|None
            The path to the CID file
        """
        if not cid_file:
            cid_file = config.get('CID_PATH', None, False)

        self.dataset_name = None
        self.data_path = None
        self.x_axis = 'x'
        self.y_axis = 'y'
        self.time_axis = 'time'
        self.precip_var = 'precip'

        self.domain = Domain(cid_file)
        self.resolution = None
        self.time_step = None
        self.tmp_dir = Path(config.get('TMP_DIR'))

    def set_data_path(self, data_path):
        """
        Set the path to the precipitation data.

        Parameters
        ----------
        data_path: str
            The path to the precipitation data
        """
        self.data_path = data_path

    def prepare_data(self):
        raise NotImplementedError("This method must be implemented in the child class.")

    def get_x_axis_for_bounds(self, x_min, x_max):
        """
        Get the x-axis slice for the given bounds.

        Parameters
        ----------
        x_min: float
            The minimum x coordinate
        x_max: float
            The maximum x coordinate

        Returns
        -------
        slice
            The slice for the x-axis
        """
        x_axis = self.domain.cids['xs'][0, :]

        return x_axis[(x_axis >= x_min) & (x_axis <= x_max)]

    def get_y_axis_for_bounds(self, y_min, y_max):
        """
        Get the y-axis slice for the given bounds.

        Parameters
        ----------
        y_min: float
            The minimum y coordinate
        y_max: float
            The maximum y coordinate

        Returns
        -------
        slice
            The slice for the y-axis
        """
        y_axis = self.domain.cids['ys'][:, 0]

        return y_axis[(y_axis >= y_min) & (y_axis <= y_max)]
