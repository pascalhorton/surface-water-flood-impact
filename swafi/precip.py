"""
Class to handle the precipitation data.
"""
import pickle
import hashlib
import dask
import pandas as pd
from pathlib import Path

from .config import Config
from .domain import Domain

config = Config()


class Precipitation:
    def __init__(self, cid_file=None, data_path=None):
        """
        The generic Precipitation class. Must be netCDF files as relies on xarray.

        Parameters
        ----------
        cid_file: str|None
            The path to the CID file
        data_path: str|None
            The path to the data files
        """
        if not cid_file:
            cid_file = config.get('CID_PATH')

        self.data_path = data_path
        self.x_axis = 'x'
        self.y_axis = 'y'
        self.time_axis = 'time'
        self.precip_var = 'precip'

        self.resolution = None
        self.time_step = None
        self.data = None
        self.missing = None
        self.domain = Domain(cid_file)
        self.tmp_dir = Path(config.get('TMP_DIR'))

    def load_data(self):
        raise NotImplementedError("This method must be implemented in the child class.")

    def get_time_series(self, cid, start, end, size=1):
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
            The number of pixels to average on (default: 1x1)

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

        if size == 1:
            return dat[self.precip_var].to_numpy()

        return dat[self.precip_var].mean(dim=[self.x_axis, self.y_axis]).to_numpy()

    def compute_quantiles_cid(self, cid):
        """
        Compute the quantiles of the precipitation data for each cell ID.

        Parameters
        ----------
        cid: int
            The cell ID for which to compute the quantiles

        Returns
        -------
        np.array
            The quantiles of the precipitation data
        """
        x, y = self.domain.get_cid_coordinates(cid)
        time_series = self.data.sel({self.x_axis: x, self.y_axis: y})
        time_series = time_series.load()

        # Compute the ranks
        quantiles = time_series.rank(dim='time', pct=True)

        return quantiles

    def _compute_hash_precip_full_data(self):
        tag_data = (
                pickle.dumps(self.data_path) +
                pickle.dumps(self.resolution) +
                pickle.dumps(self.time_step) +
                pickle.dumps(self.data['x']) +
                pickle.dumps(self.data['y']) +
                pickle.dumps(self.data['time'][0]) +
                pickle.dumps(self.data['time'][-1]) +
                pickle.dumps(self.data['precip'].shape))

        return hashlib.md5(tag_data).hexdigest()

    def _load_from_pickle(self):
        hash_tag = self._compute_hash_precip_full_data()
        filename = f"precip_full_{hash_tag}.pickle"
        tmp_filename = self.tmp_dir / filename

        if tmp_filename.exists():
            print("Precipitation already preloaded. Loading from pickle file.")
            with open(tmp_filename, 'rb') as f:
                self.data = pickle.load(f)
        else:
            print("Loading data from original files.")
            self._fill_missing_values()
            self._resample()
            self.data['time'] = pd.to_datetime(self.data['time'])

            # Save the precipitation
            with open(tmp_filename, 'wb') as f:
                pickle.dump(self.data, f)

    def _fill_missing_values(self):
        # Create a complete time series index with hourly frequency
        data_start = self.data.time.values[0]
        data_end = self.data.time.values[-1]
        complete_time_index = pd.date_range(
            start=data_start, end=data_end, freq='h')

        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            # Reindex the data to the complete time series index
            self.data = self.data.reindex(time=complete_time_index)

            # Interpolate missing values
            self.data = self.data.chunk({'time': -1})
            self.data = self.data.interpolate_na(dim='time', method='linear')

    def _resample(self):
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            # Adapt the spatial resolution
            if self.resolution != 1:
                self.data = self.data.coarsen(
                    x=self.resolution,
                    y=self.resolution,
                    boundary='trim'
                ).mean()

            # Aggregate the precipitation at the desired time step
            if self.time_step != 1:
                self.data = self.data.resample(
                    time=f'{self.time_step}h',
                ).sum(dim='time')
