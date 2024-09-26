"""
Class to handle the precipitation forecast data.
"""
from .config import Config
from .precip import Precipitation

config = Config()


class PrecipitationForecast(Precipitation):
    def __init__(self, cid_file=None):
        """
        The generic PrecipitationForecast class.

        Parameters
        ----------
        cid_file: str|None
            The path to the CID file
        """
        super().__init__(cid_file)

    def prepare_data(self):
        raise NotImplementedError("This method must be implemented in the child class.")

    def standardize(self, mean, std):
        """
        Standardize the precipitation data.

        Parameters
        ----------
        mean: float
            The mean value
        std: float
            The standard deviation
        """
        raise NotImplementedError("This method must be implemented.")

    def normalize(self, q99):
        """
        Normalize the precipitation data.

        Parameters
        ----------
        q99: float
            The 99th quantile
        """
        raise NotImplementedError("This method must be implemented.")

    def log_transform(self):
        """
        Log-transform the precipitation data.
        """
        # Use (np.log(precip + 0.1)).astype('float32')
        raise NotImplementedError("This method must be implemented.")
