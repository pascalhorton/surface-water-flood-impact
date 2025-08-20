"""
Class to handle the precipitation archive data.
"""
import pickle
import hashlib
import dask
import gc
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from tqdm import tqdm

from .config import Config
from .precip import Precipitation

config = Config()


class PrecipitationArchive(Precipitation):
    def __init__(self, year_start=None, year_end=None, cid_file=None):
        """
        The generic PrecipitationArchive class.
        Must be netCDF files as relies on xarray.

        Parameters
        ----------
        year_start: int
            The start year of the data
        year_end: int
            The end year of the data
        cid_file: str|None
            The path to the CID file
        """
        super().__init__(cid_file)

        self.year_start = year_start
        self.year_end = year_end
        if year_start is not None or year_end is not None:
            self.time_index = pd.date_range(start=f'{year_start}-01-01',
                                            end=f'{year_end}-12-31', freq='MS')
        self.missing = None

        self.hash_tag = None
        self.pickle_files = []
        self.cid_time_series = None
        self.mem_nb_pixels = 64  # Number of pixels to process at once (per spatial dimension; e.g. 100x100)

    def reset(self):
        """
        Reset the data.
        """
        self.hash_tag = None
        self.pickle_files = []
        self.cid_time_series = None

    def prepare_data(self):
        raise NotImplementedError("This method must be implemented in the child class.")

    def get_time_series(self, cid, start, end, size=1, as_xr=False):
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
        as_xr: bool
            Return as xarray dataset (default: False)

        Returns
        -------
        np.array
            The timeseries as a numpy array
        """
        x, y = self.domain.get_cid_coordinates(cid)
        dx = self.domain.resolution[0]
        dy = self.domain.resolution[1]
        dpx = (size - 1) / 2

        if len(self.pickle_files) == 0:
            raise ValueError("The precipitation data must be first pre-loaded.")

        ts = []

        for f in self.pickle_files:
            try:
                with open(f, 'rb') as file:
                    data = pickle.load(file)
                    data = data.chunk({'time': -1, self.x_axis: 'auto',
                                       self.y_axis: 'auto'})  # Chunk the data
                    dat = data.sel(
                        {self.x_axis: slice(x - dx * dpx, x + dx * dpx),
                         self.y_axis: slice(y + dy * dpx, y - dy * dpx)
                         }).compute()  # Compute only the selected data

                    ts.append(dat)
                    del data
                    gc.collect()
            except EOFError:
                raise EOFError(f"Error: {f} is empty or corrupted.")

        if len(ts) == 0:
            raise ValueError(f"No data found for CID {cid}")

        ts = xr.concat(ts, dim=self.time_axis)
        ts = ts.sel({self.time_axis: slice(start, end)})

        if as_xr:
            return ts

        if size == 1:
            return ts[self.precip_var].to_numpy()

        return ts[self.precip_var].mean(dim=[self.x_axis, self.y_axis]).to_numpy()

    def preload_all_cid_data(self, cids):
        """
        Preload all the data for each cell ID.

        Parameters
        ----------
        cids: list
            The list of cell IDs
        """
        self.cid_time_series = None  # Necessary to reset the data !
        hash_tag = hashlib.md5(
            pickle.dumps(self.pickle_files) + pickle.dumps(cids)).hexdigest()

        filename = f"precip_{self.dataset_name.lower()}_all_cids_{hash_tag}.pickle"
        tmp_filename = self.tmp_dir / filename

        if tmp_filename.exists():
            print("Loading all data for each CID from pickle file.")
            try:
                with open(tmp_filename, 'rb') as f:
                    self.cid_time_series = pickle.load(f)
                return

            except EOFError:
                raise EOFError(f"Error: {tmp_filename} is empty or corrupted.")

        locations = [self.domain.get_cid_coordinates(cid) for cid in cids]

        for idx in tqdm(range(len(self.time_index)),
                        desc="Preloading all data for each CID"):
            f = self.pickle_files[idx]
            try:
                with open(f, 'rb') as file:
                    ts = []
                    data = pickle.load(file)
                    for x, y in locations:
                        try:
                            dat = data[self.precip_var].sel(
                                {self.x_axis: x, self.y_axis: y}
                            )
                        except ValueError as e:
                            print(e)
                            print(f"Error with file {f} and location {x}, {y}")
                            print(f"Data shape: {data[self.precip_var].shape}")
                            data.info()

                        ts.append(dat)

                    ts_xr = xr.concat(ts, dim='cid')
                    if self.cid_time_series is None:
                        self.cid_time_series = ts_xr
                    else:
                        self.cid_time_series = xr.concat([self.cid_time_series, ts_xr],
                                                         dim=self.time_axis)

            except EOFError:
                raise EOFError(f"Error: {f} is empty or corrupted.")

        self.cid_time_series['cid'] = cids

        with open(tmp_filename, 'wb') as f:
            pickle.dump(self.cid_time_series, f)

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

    def generate_pickles_for_subdomain(self, x_axis, y_axis):
        """
        Generate pickle files for the subdomain defined by the x and y axes.

        Parameters
        ----------
        x_axis: slice|np.array
            The slice for the x axis
        y_axis: slice|np.array
            The slice for the y axis
        """
        if not isinstance(x_axis, np.ndarray):
            x_axis = x_axis.to_numpy()
        if not isinstance(y_axis, np.ndarray):
            y_axis = y_axis.to_numpy()

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

            try:
                with open(original_file, 'rb') as f_in:
                    data = pickle.load(f_in)
                    data = data.sel({self.x_axis: x_axis, self.y_axis: y_axis})

                    # If the array is smaller than the expected size, fill with NaN
                    if data[self.precip_var].shape[1:] != (len(y_axis), len(x_axis)):
                        expected_shape = (
                            len(data[self.time_axis]),
                            len(y_axis),
                            len(x_axis)
                        )

                        print(f"Filling missing values for {t.year}-{t.month:02} "
                              f"with NaN in {self.precip_var} variable. "
                              f"Expected shape: {expected_shape}, "
                              f"actual shape: {data[self.precip_var].shape}")

                        # Create an array filled with np.nan of the expected shape
                        filled_data = np.full(expected_shape, np.nan, dtype='float32')

                        # Get the available x and y coordinates in the data
                        data_x = data[self.x_axis].values
                        data_y = data[self.y_axis].values

                        # Find the intersection indices for x and y
                        x_idx = [i for i, x in enumerate(x_axis) if x in data_x]
                        y_idx = [i for i, y in enumerate(y_axis) if y in data_y]

                        # Find the corresponding indices in the data
                        data_x_i = [np.where(data_x == x_axis[i])[0][0] for i in x_idx]
                        data_y_i = [np.where(data_y == y_axis[i])[0][0] for i in y_idx]

                        # Place the available data into the correct positions
                        for i in range(len(data_x_i)):
                            for j in range(len(data_y_i)):
                                x_i = data_x_i[i]
                                y_i = data_y_i[j]
                                filled_data[:, y_idx[j], x_idx[i]] = \
                                    data[self.precip_var].values[:, y_i, x_i]

                        # Assign the filled data back to the xarray DataArray
                        data[self.precip_var] = filled_data

                    with open(tmp_filename, 'wb') as f_out:
                        pickle.dump(data, f_out)

            except EOFError:
                raise EOFError(f"Error: {original_file} is empty or corrupted.")

    def standardize(self, mean, std):
        """
        Standardize the precipitation data.

        Parameters
        ----------
        mean: np.array
            The mean values (per pixel)
        std: np.array
            The standard deviations (per pixel)
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

            try:
                with open(original_file, 'rb') as f_in:
                    data = pickle.load(f_in)
                    precip = data[self.precip_var]
                    data[self.precip_var] = ((precip - mean) / std).astype('float32')

                    with open(tmp_filename, 'wb') as f_out:
                        pickle.dump(data, f_out)

            except EOFError:
                raise EOFError(f"Error: {original_file} is empty or corrupted.")

    def normalize(self, q99):
        """
        Normalize the precipitation data.

        Parameters
        ----------
        q99: np.array
            The 99th quantile (per pixel)
        """
        # Add dimension to q99
        q99 = np.expand_dims(q99, axis=0)

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

            try:
                with open(original_file, 'rb') as f_in:
                    data = pickle.load(f_in)
                    precip = data[self.precip_var]
                    min_precip = float(precip.min())  # Might not be 0 when log-transformed

                    data[self.precip_var] = ((precip - min_precip) / (q99 - min_precip)).astype('float32')

                    with open(tmp_filename, 'wb') as f_out:
                        pickle.dump(data, f_out)

            except EOFError:
                raise EOFError(f"Error: {original_file} is empty or corrupted.")

    def log_transform(self):
        """
        Log-transform the precipitation data.
        """
        if not self.hash_tag.startswith("log_"):
            self.hash_tag = "log_" + self.hash_tag

        for idx in tqdm(range(len(self.time_index)),
                        desc="Log-transforming precipitation data"):
            original_file = self.pickle_files[idx]
            t = self.time_index[idx]
            filename = (f"precip_{self.dataset_name.lower()}_subdomain_{t.year}-"
                        f"{t.month:02}_{self.hash_tag}.pickle")
            tmp_filename = self.tmp_dir / filename
            self.pickle_files[idx] = tmp_filename

            if tmp_filename.exists():
                continue

            try:
                with open(original_file, 'rb') as f_in:
                    data = pickle.load(f_in)
                    precip = data[self.precip_var]
                    data[self.precip_var] = (np.log1p(precip)).astype('float32')

                    with open(tmp_filename, 'wb') as f_out:
                        pickle.dump(data, f_out)

            except EOFError:
                raise EOFError(f"Error: {original_file} is empty or corrupted.")

            except ValueError as e:
                raise ValueError(f"Error with file {original_file} at time {t}: {e}")

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
            try:
                with open(mean_file, 'rb') as f:
                    mean = pickle.load(f)
            except EOFError:
                raise EOFError(f"Error: {mean_file} is empty or corrupted.")

            try:
                with open(std_file, 'rb') as f:
                    std = pickle.load(f)
            except EOFError:
                raise EOFError(f"Error: {std_file} is empty or corrupted.")

            return mean, std

        # Open first precipitation file to get the dimensions
        try:
            with open(self.pickle_files[0], 'rb') as f_in:
                data = pickle.load(f_in)
                n_rows, n_cols = data[self.precip_var].shape[1:]
        except EOFError:
            raise EOFError(f"Error: {self.pickle_files[0]} is empty or corrupted.")

        # Compute mean and standard deviation by spatial chunks (for memory efficiency)
        mean = np.zeros((n_rows, n_cols))
        std = np.zeros((n_rows, n_cols))
        for i in tqdm(np.arange(0, n_rows + 1, self.mem_nb_pixels),
                      desc="Computing mean and standard deviation per pixel"):
            for j in np.arange(0, n_cols + 1, self.mem_nb_pixels):
                x_size = min(self.mem_nb_pixels, n_rows - i)
                y_size = min(self.mem_nb_pixels, n_cols - j)
                data = self.get_spatial_chunk_data(i, j, x_size, y_size)
                mean[i:i + x_size, j:j + y_size] = np.nanmean(data, axis=0)
                std[i:i + x_size, j:j + y_size] = np.nanstd(data, axis=0)

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
            try:
                with open(tmp_filename, 'rb') as f:
                    quantiles = pickle.load(f)
            except EOFError:
                raise EOFError(f"Error: {tmp_filename} is empty or corrupted.")

            return quantiles

        # Open first precipitation file to get the dimensions
        try:
            with open(self.pickle_files[0], 'rb') as f_in:
                data = pickle.load(f_in)
                n_rows, n_cols = data[self.precip_var].shape[1:]
        except EOFError:
            raise EOFError(f"Error: {self.pickle_files[0]} is empty or corrupted.")

        # Compute quantile by spatial chunks (for memory efficiency)
        quantiles = np.zeros((n_rows, n_cols))
        for i in tqdm(np.arange(0, n_rows + 1, self.mem_nb_pixels),
                      desc=f"Computing {quantile} quantile per pixel"):
            for j in np.arange(0, n_cols + 1, self.mem_nb_pixels):
                x_size = min(self.mem_nb_pixels, n_rows - i)
                y_size = min(self.mem_nb_pixels, n_cols - j)
                data = self.get_spatial_chunk_data(i, j, x_size, y_size)
                quantiles[i:i + x_size, j:j + y_size] = np.nanquantile(data, quantile, axis=0)
                del data
                gc.collect()

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

            try:
                with open(original_file, 'rb') as f_in:
                    data_file = pickle.load(f_in)
                    data_file = data_file[self.precip_var].values
                    data_file = data_file[:, i:i + x_size, j:j + y_size]
                    if data_chunk is None:
                        data_chunk = data_file
                    else:
                        data_chunk = np.concatenate((data_chunk, data_file), axis=0)
            except EOFError:
                raise EOFError(f"Error: {original_file} is empty or corrupted.")
            except ValueError as e:
                raise ValueError(f"Error with file {original_file} at indices "
                                 f"{i}:{i + x_size}, {j}:{j + y_size}: {e}")

        return data_chunk

    def get_data_chunk(self, t_start, t_end, x_start, x_end, y_start, y_end, cid=None):
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
        cid: int|None
            The cell ID (default: None)

        Returns
        -------
        np.array
            The precipitation data for the temporal and spatial chunk
        """
        if self.cid_time_series is not None and cid is not None:
            try:
                ts = self.cid_time_series.sel(
                    time=slice(t_start, t_end),
                    cid=cid
                ).to_numpy()
            except KeyError as e:
                print(f"Error with CID {cid} and time {t_start} to {t_end}")
                print(f"File: precip_{self.dataset_name.lower()}_all_cids_[hash].pickle")
                print(e)

            # If the time series is 1D, add 2 dimensions
            if len(ts.shape) == 1:
                ts = np.expand_dims(ts, axis=(1, 2))

            return ts

        # Get the index/indices in the temporal index
        try:
            idx_start = self.time_index.get_loc(t_start.normalize().replace(day=1))
        except KeyError:
            idx_start = 0
        try:
            idx_end = self.time_index.get_loc(t_end.normalize().replace(day=1))
        except KeyError:
            idx_end = len(self.time_index) - 1

        data = None
        for idx in range(idx_start, idx_end + 1):
            pk_file = self.pickle_files[idx]
            try:
                with open(pk_file, 'rb') as f_in:
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
            except EOFError:
                raise EOFError(f"Error: {pk_file} is empty or corrupted.")

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
        self.hash_tag = self._compute_hash_precip_full_data(
            data['x'].to_numpy(), data['y'].to_numpy())

        data['time'] = pd.to_datetime(data['time'])

        for idx in tqdm(
                range(len(self.time_index)),
                desc="Generating pickle files for precipitation data"
        ):
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
            subset = self._remove_duplicate_timestamps(subset)
            subset = self._fill_missing_values(subset)
            subset = self._resample(subset)
            subset = subset.compute()

            # Check for duplicates
            if len(subset['time'].values) != len(np.unique(subset['time'].values)):
                raise ValueError(
                    f"Duplicate timestamps found in subset for {t.year}-{t.month:02}")

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
    def _remove_duplicate_timestamps(data):
        # Identify duplicate timestamps
        _, index = np.unique(data['time'], return_index=True)
        unique_times = data['time'].values[index]

        # Reindex the dataset to remove duplicates
        data = data.sel(time=unique_times)

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

        # Replace NaN values with 0
        data = data.fillna(0)

        return data
