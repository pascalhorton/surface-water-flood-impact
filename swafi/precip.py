"""
Class to handle the precipitation data.
"""
import pickle
import hashlib
import dask
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm

from .config import Config
from .domain import Domain

config = Config()


class Precipitation:
    def __init__(self, year_start, year_end, cid_file=None):
        """
        The generic Precipitation class. Must be netCDF files as relies on xarray.

        Parameters
        ----------
        year_start: int
            The start year of the data
        year_end: int
            The end year of the data
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
        self.year_start = year_start
        self.year_end = year_end
        self.time_index = pd.date_range(start=f'{year_start}-01-01',
                                        end=f'{year_end}-12-31', freq='MS')
        self.missing = None
        self.tmp_dir = Path(config.get('TMP_DIR'))

        self.pickle_files_full = []
        self.pickle_files_subdomain = []

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

        ts = []

        for f in self.pickle_files_full:
            with open(f, 'rb') as file:
                data = pickle.load(file)
                dat = data.sel(
                    {self.x_axis: slice(x - dx * dpx, x + dx * dpx),
                     self.y_axis: slice(y + dy * dpx, y - dy * dpx)
                     })
                ts.append(dat)

        ts = xr.concat(ts, dim=self.time_axis)
        ts = ts.sel({self.time_axis: slice(start, end)})

        if size == 1:
            return ts[self.precip_var].to_numpy()

        return ts[self.precip_var].mean(dim=[self.x_axis, self.y_axis]).to_numpy()

    def save_nc_file_per_cid(self, cid, start, end):
        """
        Save the precipitation time series for the given cell ID and the given
        period (between start and end) in a netCDF file.

        Parameters
        ----------
        cid: int
            The cell ID
        start: datetime.datetime|str
            The start of the period to extract
        end: datetime.datetime|str
            The end of the period to extract
        """
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        y_start = start.year
        y_end = end.year

        hash_tag = self._compute_hash_single_cid(y_start, y_end)
        filename = f"precip_cid_{cid}_{hash_tag}.nc"
        tmp_filename = self.tmp_dir / filename

        if tmp_filename.exists():
            return

        time_series = self.get_time_series(cid, start, end)
        time_series.to_netcdf(tmp_filename)

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
        # Use pickles to store the data
        hash_tag_pk = self._compute_hash_precip_full_data()
        filename_pk = f"precip_quantiles_cid_{cid}_{hash_tag_pk}.pickle"
        tmp_filename_pk = self.tmp_dir / filename_pk

        if tmp_filename_pk.exists():
            print(f"Loading quantiles for CID {cid} from pickle file.")
            with open(tmp_filename_pk, 'rb') as f:
                quantiles = pickle.load(f)
                return quantiles

        # Check if netCDF file of the time series exists
        hash_tag_nc = self._compute_hash_single_cid(self.year_start, self.year_end)
        filename_nc = f"precip_cid_{cid}_{hash_tag_nc}.nc"
        tmp_filename_nc = self.tmp_dir / filename_nc

        # Extract the time series
        if tmp_filename_nc.exists():
            print(f"Loading time series for CID {cid} from netCDF file.")
            time_series = xr.open_dataset(tmp_filename_nc)
        else:
            start = pd.to_datetime(f'{self.year_start}-01-01 00:00:00')
            end = pd.to_datetime(f'{self.year_end}-12-31 23:59:59')
            time_series = self.get_time_series(cid, start, end)
            time_series = time_series.load()
            time_series.to_netcdf(tmp_filename_nc)

        # Compute the ranks
        quantiles = time_series.rank(dim='time', pct=True)

        # Save as pickle
        with open(tmp_filename_pk, 'wb') as f:
            pickle.dump(quantiles, f)

        return quantiles

    def _compute_hash_precip_full_data(self, x_axis=None, y_axis=None):
        tag_data = (
                pickle.dumps(self.dataset_name) +
                pickle.dumps(self.resolution) +
                pickle.dumps(self.time_step) +
                pickle.dumps(x_axis) +
                pickle.dumps(y_axis))

        return hashlib.md5(tag_data).hexdigest()

    def _compute_hash_single_cid(self, y_start, y_end):
        tag_data = (
                pickle.dumps(self.dataset_name) +
                pickle.dumps(self.resolution) +
                pickle.dumps(self.time_step) +
                pickle.dumps(y_start) +
                pickle.dumps(y_end))

        return hashlib.md5(tag_data).hexdigest()

    def _generate_pickle_files(self, data):
        hash_tag = self._compute_hash_precip_full_data(data['x'], data['y'])

        data['time'] = pd.to_datetime(data['time'])

        for idx in tqdm(range(len(self.time_index)),
                        desc="Generating pickle files for precipitation data"):
            t = self.time_index[idx]
            filename = f"precip_full_{t.year}_{t.month}_{hash_tag}.pickle"
            tmp_filename = self.tmp_dir / filename
            self.pickle_files_full.append(tmp_filename)

            if tmp_filename.exists():
                continue

            end_time = (t + pd.offsets.MonthEnd(0)).replace(
                hour=23, minute=59, second=59)
            subset = data.sel(time=slice(t, end_time)).compute()
            subset = self._fill_missing_values(subset)
            subset = self._resample(subset)
            subset = subset.compute()

            assert len(subset.dims) == 3, "Precipitation must be 3D"

            with open(tmp_filename, 'wb') as f:
                pickle.dump(subset, f)

    def generate_pickles_for_subdomain(self, x_axis, y_axis):
        hash_tag = self._compute_hash_precip_full_data(x_axis, y_axis)

        for idx in tqdm(range(len(self.time_index)),
                        desc="Generating pickle files for subdomain"):
            t = self.time_index[idx]
            filename = f"precip_subdomain_{t.year}_{t.month}_{hash_tag}.pickle"
            tmp_filename = self.tmp_dir / filename
            self.pickle_files_subdomain.append(tmp_filename)

            if tmp_filename.exists():
                continue

            with open(self.pickle_files_full[idx], 'rb') as f:
                data = pickle.load(f)
                data = data.sel({self.x_axis: x_axis, self.y_axis: y_axis})

                with open(tmp_filename, 'wb') as f:
                    pickle.dump(data, f)

    def _resample(self, data):
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            # Adapt the spatial resolution
            if self.resolution != 1:
                data = data.coarsen(
                    x=self.resolution,
                    y=self.resolution,
                    boundary='trim'
                ).mean()

            # Aggregate the precipitation at the desired time step
            if self.time_step != 1:
                data = data.resample(
                    time=f'{self.time_step}h',
                ).sum(dim='time')

        return data

    @staticmethod
    def _fill_missing_values(data):
        # Create a complete time series index with hourly frequency
        data_start = data.time.values[0]
        data_start = pd.Timestamp(data_start).replace(day=1, hour=0)
        data_end = (data_start + pd.offsets.MonthEnd(0)).replace(hour=23)
        complete_time_index = pd.date_range(
            start=data_start, end=data_end, freq='h')

        if len(complete_time_index) != len(data.time):
            with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                # Reindex the data to the complete time series index
                data = data.reindex(time=complete_time_index)

                # Interpolate missing values
                data = data.chunk({'time': -1})
                data = data.interpolate_na(dim='time', method='linear')

        return data
