"""
Class to handle the precipitation data.
"""
import pickle
import hashlib
import dask
import numpy as np
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

        self.hash_tag = None
        self.pickle_files = []
        self.mem_nb_pixels = 20  # Number of pixels to process at once (per spatial dimension; e.g. 100x100)

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
        Get the x axis slice for the given bounds.

        Parameters
        ----------
        x_min: float
            The minimum x coordinate
        x_max: float
            The maximum x coordinate

        Returns
        -------
        slice
            The slice for the x axis
        """
        x_axis = self.domain.cids['xs'][0, :]

        return x_axis[(x_axis >= x_min) & (x_axis <= x_max)]

    def get_y_axis_for_bounds(self, y_min, y_max):
        """
        Get the y axis slice for the given bounds.

        Parameters
        ----------
        y_min: float
            The minimum y coordinate
        y_max: float
            The maximum y coordinate

        Returns
        -------
        slice
            The slice for the y axis
        """
        y_axis = self.domain.cids['ys'][:, 0]

        return y_axis[(y_axis >= y_min) & (y_axis <= y_max)]

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

        for f in self.pickle_files:
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

        self.hash_tag = self._compute_hash_single_cid(y_start, y_end)
        filename = f"precip_{self.dataset_name.lower()}_cid_{self.hash_tag}.nc"
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
        filename_pk = (f"precip_{self.dataset_name.lower()}_q_"
                       f"cid__{hash_tag_pk}.pickle")
        tmp_filename_pk = self.tmp_dir / filename_pk

        if tmp_filename_pk.exists():
            print(f"Loading quantiles for CID {cid} from pickle file.")
            with open(tmp_filename_pk, 'rb') as f:
                quantiles = pickle.load(f)
                return quantiles

        # Check if netCDF file of the time series exists
        hash_tag_nc = self._compute_hash_single_cid(self.year_start, self.year_end)
        filename_nc = f"precip_{self.dataset_name.lower()}_cid_{hash_tag_nc}.nc"
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

    def generate_pickles_for_subdomain(self, x_axis, y_axis):
        """
        Generate pickle files for the subdomain defined by the x and y axes.

        Parameters
        ----------
        x_axis: slice
            The slice for the x axis
        y_axis: slice
            The slice for the y axis
        """
        self.hash_tag = self._compute_hash_precip_full_data(x_axis, y_axis)

        for idx in tqdm(range(len(self.time_index)),
                        desc="Generating pickle files for subdomain"):
            original_file = self.pickle_files[idx]
            t = self.time_index[idx]
            filename = (f"precip_{self.dataset_name.lower()}_subdomain_{t.year}-"
                        f"{t.month:02}_{self.hash_tag}.pickle")
            tmp_filename = self.tmp_dir / filename
            self.pickle_files[idx] = tmp_filename

            if tmp_filename.exists():
                continue

            with open(original_file, 'rb') as f_in:
                data = pickle.load(f_in)
                data = data.sel({self.x_axis: x_axis, self.y_axis: y_axis})

                with open(tmp_filename, 'wb') as f_out:
                    pickle.dump(data, f_out)

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
        for idx in tqdm(range(len(self.time_index)),
                        desc="Standardizing precipitation data"):
            original_file = self.pickle_files[idx]
            t = self.time_index[idx]
            filename = (f"precip_{self.dataset_name.lower()}_standardized_{t.year}-"
                        f"{t.month:02}_{self.hash_tag}.pickle")
            tmp_filename = self.tmp_dir / filename
            self.pickle_files[idx] = tmp_filename

            if tmp_filename.exists():
                continue

            with open(original_file, 'rb') as f_in:
                data = pickle.load(f_in)
                data = data.assign({
                    self.precip_var: ((data[self.precip_var] - mean) / std).astype('float32')})

                with open(tmp_filename, 'wb') as f_out:
                    pickle.dump(data, f_out)

    def normalize(self, q95):
        """
        Normalize the precipitation data.

        Parameters
        ----------
        q95: float
            The 95th quantile
        """
        for idx in tqdm(range(len(self.time_index)),
                        desc="Normalizing precipitation data"):
            original_file = self.pickle_files[idx]
            t = self.time_index[idx]
            filename = (f"precip_{self.dataset_name.lower()}_normalized_{t.year}-"
                        f"{t.month:02}_{self.hash_tag}.pickle")
            tmp_filename = self.tmp_dir / filename
            self.pickle_files[idx] = tmp_filename

            if tmp_filename.exists():
                continue

            with open(original_file, 'rb') as f_in:
                data = pickle.load(f_in)
                data = data.assign({
                    self.precip_var: (data[self.precip_var] / q95).astype('float32')})

                with open(tmp_filename, 'wb') as f_out:
                    pickle.dump(data, f_out)

    def log_transform(self):
        """
        Log-transform the precipitation data.
        """
        for idx in tqdm(range(len(self.time_index)),
                        desc="Log-transforming precipitation data"):
            original_file = self.pickle_files[idx]
            t = self.time_index[idx]
            filename = (f"precip_{self.dataset_name.lower()}_log_{t.year}-"
                        f"{t.month:02}_{self.hash_tag}.pickle")
            tmp_filename = self.tmp_dir / filename
            self.pickle_files[idx] = tmp_filename

            if tmp_filename.exists():
                continue

            with open(original_file, 'rb') as f_in:
                data = pickle.load(f_in)
                data = data.assign({self.precip_var: np.log(data[self.precip_var] + 1e-10)})

                with open(tmp_filename, 'wb') as f_out:
                    pickle.dump(data, f_out)

    def compute_mean_and_std_per_pixel(self):
        """
        Compute the mean and standard deviation of the precipitation data for each pixel.

        Returns
        -------
        np.array, np.array
            The mean and standard deviation of the precipitation data
        """
        # Compute hash tag by hashing the pickle files list
        hash_tag = hashlib.md5(pickle.dumps(self.pickle_files)).hexdigest()
        filename_mean = f"precip_{self.dataset_name.lower()}_mean_{hash_tag}.pickle"
        filename_std = f"precip_{self.dataset_name.lower()}_std_{hash_tag}.pickle"

        # If the files already exist, load them
        mean_file = self.tmp_dir / filename_mean
        std_file = self.tmp_dir / filename_std
        if mean_file.exists() and std_file.exists():
            with open(mean_file, 'rb') as f:
                mean = pickle.load(f)
            with open(std_file, 'rb') as f:
                std = pickle.load(f)
            return mean, std

        # Open first precipitation file to get the dimensions
        with open(self.pickle_files[0], 'rb') as f_in:
            data = pickle.load(f_in)
            n_rows, n_cols = data[self.precip_var].shape[1:]

        # Compute mean and standard deviation by spatial chunks (for memory efficiency)
        mean = np.zeros((n_rows, n_cols))
        std = np.zeros((n_rows, n_cols))
        for i in tqdm(np.arange(0, n_rows + 1, self.mem_nb_pixels),
                      desc="Computing mean and standard deviation per pixel"):
            for j in np.arange(0, n_cols + 1, self.mem_nb_pixels):
                x_size = min(self.mem_nb_pixels, n_rows - i)
                y_size = min(self.mem_nb_pixels, n_cols - j)
                data = self.get_spatial_chunk_data(i, j, x_size, y_size)
                mean[i:i + x_size, j:j + y_size] = np.mean(data, axis=0)
                std[i:i + x_size, j:j + y_size] = np.std(data, axis=0)

        # Save mean and standard deviation
        with open(mean_file, 'wb') as f:
            pickle.dump(mean, f)
        with open(std_file, 'wb') as f:
            pickle.dump(std, f)

        return mean, std

    def compute_quantile_per_pixel(self, quantile):
        """
        Compute the quantile of the precipitation data for each pixel.

        Parameters
        ----------
        quantile: float
            The quantile to compute

        Returns
        -------
        np.array
            The quantile of the precipitation data
        """
        # Compute hash tag by hashing the pickle files list
        hash_tag = hashlib.md5(pickle.dumps(self.pickle_files)).hexdigest()
        filename = f"precip_{self.dataset_name.lower()}_q_{quantile:.3f}_{hash_tag}.pickle"

        # If the file already exists, load it
        tmp_filename = self.tmp_dir / filename
        if tmp_filename.exists():
            with open(tmp_filename, 'rb') as f:
                quantiles = pickle.load(f)
            return quantiles

        # Open first precipitation file to get the dimensions
        with open(self.pickle_files[0], 'rb') as f_in:
            data = pickle.load(f_in)
            n_rows, n_cols = data[self.precip_var].shape[1:]

        # Compute quantile by spatial chunks (for memory efficiency)
        quantiles = np.zeros((n_rows, n_cols))
        for i in tqdm(np.arange(0, n_rows + 1, self.mem_nb_pixels),
                      desc=f"Computing {quantile} quantile per pixel"):
            for j in np.arange(0, n_cols + 1, self.mem_nb_pixels):
                x_size = min(self.mem_nb_pixels, n_rows - i)
                y_size = min(self.mem_nb_pixels, n_cols - j)
                data = self.get_spatial_chunk_data(i, j, x_size, y_size)
                quantiles[i:i + x_size, j:j + y_size] = np.quantile(data, quantile, axis=0)

        # Save quantile
        with open(tmp_filename, 'wb') as f:
            pickle.dump(quantiles, f)

        return quantiles

    def get_spatial_chunk_data(self, i, j, x_size, y_size):
        """
        Get the precipitation data for a spatial chunk.

        Parameters
        ----------
        i: int
            The starting row index
        j: int
            The starting column index
        x_size: int
            The number of rows
        y_size: int
            The number of columns

        Returns
        -------
        np.array
            The precipitation data for the spatial chunk
        """
        data_chunk = None
        for idx in range(len(self.time_index)):
            original_file = self.pickle_files[idx]

            with open(original_file, 'rb') as f_in:
                data_file = pickle.load(f_in)
                data_file = data_file[self.precip_var].values
                data_file = data_file[:, i:i + x_size, j:j + y_size]
                if data_chunk is None:
                    data_chunk = data_file
                else:
                    data_chunk = np.concatenate((data_chunk, data_file))

        return data_chunk

    def get_data_chunk(self, t_start, t_end, x_start, x_end, y_start, y_end):
        """
        Get the precipitation data for a temporal and spatial chunk.

        Parameters
        ----------
        t_start:
            The starting time
        t_end:
            The ending time
        x_start: float
            The starting x coordinate
        x_end: float
            The ending x coordinate
        y_start: float
            The starting y coordinate
        y_end: float
            The ending y coordinate

        Returns
        -------
        np.array
            The precipitation data for the temporal and spatial chunk
        """
        # Get the index/indices in the temporal index
        idx_start = self.time_index.get_loc(t_start.normalize().replace(day=1))
        idx_end = self.time_index.get_loc(t_end.normalize().replace(day=1))

        data = None
        for idx in range(idx_start, idx_end + 1):
            with open(self.pickle_files[idx], 'rb') as f_in:
                data_file = pickle.load(f_in)
                data_file = data_file[self.precip_var].sel(
                    time=slice(t_start, t_end),
                    x=slice(x_start, x_end),
                    y=slice(y_start, y_end)
                ).to_numpy()

                if data is None:
                    data = data_file
                else:
                    data = np.concatenate((data, data_file))

        return data

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
        self.hash_tag = self._compute_hash_precip_full_data(data['x'], data['y'])

        data['time'] = pd.to_datetime(data['time'])

        for idx in tqdm(range(len(self.time_index)),
                        desc="Generating pickle files for precipitation data"):
            t = self.time_index[idx]
            filename = (f"precip_{self.dataset_name.lower()}_full_{t.year}-"
                        f"{t.month:02}_{self.hash_tag}.pickle")
            tmp_filename = self.tmp_dir / filename
            self.pickle_files.append(tmp_filename)

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
