"""
Class to handle the precipitation archive data.
"""
import hashlib
import logging
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .config import Config
from .precip import Precipitation

config = Config()

logger = logging.getLogger(__name__)


class PrecipitationArchive(Precipitation):
    def __init__(self, year_start=None, year_end=None, cid_file=None):
        """
        The generic PrecipitationArchive class. The data is backed by a zarr
        store (see the build_zarr_store() methods of the subclasses) and opened
        lazily: reads only touch the chunks covering the selection.

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

        self.cid_time_series = None
        self.full_grid_data = None
        self.native_time_step = 1  # Native time step of the source [h]
        self.mem_nb_pixels = 64  # Number of pixels to process at once (per spatial dimension; e.g. 100x100)
        self._transform_tag = ''  # Applied lazy transforms (part of cache keys)

    def reset(self):
        """
        Reset the data.
        """
        self.data = None
        self.cid_time_series = None
        self.full_grid_data = None
        self._transform_tag = ''

    def prepare_data(self):
        raise NotImplementedError("This method must be implemented in the child class.")

    def open_zarr(self, zarr_path):
        """
        Open a precipitation zarr store lazily. Only the chunks covering later
        selections are read from disk.

        Parameters
        ----------
        zarr_path: str|Path
            The path to the zarr store.
        """
        zarr_path = Path(zarr_path)
        if not (zarr_path / 'zarr.json').exists():
            raise FileNotFoundError(
                f"'{zarr_path}' is not an initialized zarr store.")

        self.data = xr.open_zarr(zarr_path, consolidated=False)
        self.resolution = 1
        self.time_step = self.native_time_step

        # Select the data for the given years
        if self.year_start:
            self.data = self.data.sel(time=slice(f'{self.year_start}-01-01', None))
        if self.year_end:
            self.data = self.data.sel(time=slice(None, f'{self.year_end}-12-31'))

    def preload_full_grid(self):
        """
        Load the full (lazy) dataset into memory. Optional: patch reads from the
        zarr store are cheap, but an in-memory grid makes batch generation with
        large spatial windows faster still (if it fits in RAM).
        """
        logger.info("Preloading full precipitation grid into memory...")
        self.full_grid_data = self.data.compute()
        logger.info("Full grid loaded: shape %s, size %.1f GB",
                    dict(self.full_grid_data.sizes),
                    self.full_grid_data.nbytes / 1e9)

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
        if self.data is None:
            raise ValueError("The precipitation data must be first loaded.")

        x, y = self.domain.get_cid_coordinates(cid)
        dx = self.domain.resolution[0]
        dy = self.domain.resolution[1]
        dpx = (size - 1) / 2

        source = self.full_grid_data if self.full_grid_data is not None else self.data
        ts = source.sel(
            {self.x_axis_dim: slice(x - dx * dpx, x + dx * dpx),
             self.y_axis_dim: slice(y + dy * dpx, y - dy * dpx),
             self.time_axis_dim: slice(start, end)}
        ).compute()

        if len(ts[self.time_axis_dim]) == 0:
            raise ValueError(f"No data found for CID {cid}")

        if as_xr:
            return ts

        if size == 1:
            return ts[self.precip_var].to_numpy()

        return ts[self.precip_var].mean(dim=[self.x_axis_dim, self.y_axis_dim]).to_numpy()

    def preload_all_cid_data(self, cids):
        """
        Preload the 1-D time series for each cell ID (single vectorized pass
        over the store). The result is cached in TMP_DIR as netCDF.

        Parameters
        ----------
        cids: list
            The list of cell IDs
        """
        self.cid_time_series = None  # Necessary to reset the data !
        cids = np.asarray(cids)
        hash_tag = self._compute_cache_hash(cids.tobytes())

        filename = f"precip_{self.dataset_name.lower()}_all_cids_{hash_tag}.nc"
        tmp_filename = self.tmp_dir / filename

        if tmp_filename.exists():
            logger.info("Loading all data for each CID from netCDF file.")
            self.cid_time_series = xr.load_dataarray(tmp_filename)
            return

        locations = [self.domain.get_cid_coordinates(cid) for cid in cids]
        xs = xr.DataArray([x for x, _ in locations], dims='cid')
        ys = xr.DataArray([y for _, y in locations], dims='cid')

        logger.info("Preloading all data for each CID (single pass over the store).")
        ts = self.data[self.precip_var].sel(
            {self.x_axis_dim: xs, self.y_axis_dim: ys})
        ts = ts.assign_coords(cid=('cid', cids))
        self.cid_time_series = ts.compute()

        # Check again that the file was not created in the meantime
        if tmp_filename.exists():
            return

        self.cid_time_series.to_netcdf(tmp_filename)

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

        hash_tag = self._compute_cache_hash(f"{cid}_{start}_{end}")
        filename = f"precip_{self.dataset_name.lower()}_cid_{cid}_{hash_tag}.nc"
        tmp_filename = self.tmp_dir / filename

        if tmp_filename.exists():
            return

        time_series = self.get_time_series(cid, start, end, as_xr=True)
        time_series.to_netcdf(tmp_filename)

    def select_subdomain(self, x_axis, y_axis):
        """
        Restrict the (lazy) data to the given axes. Cells of the axes outside
        the data extent are filled with NaN.

        Parameters
        ----------
        x_axis: xr.DataArray|np.array
            The x coordinates to select
        y_axis: xr.DataArray|np.array
            The y coordinates to select
        """
        if not isinstance(x_axis, np.ndarray):
            x_axis = x_axis.to_numpy()
        if not isinstance(y_axis, np.ndarray):
            y_axis = y_axis.to_numpy()

        self.data = self.data.reindex(
            {self.x_axis_dim: x_axis, self.y_axis_dim: y_axis})

    def standardize(self, mean, std):
        """
        Standardize the precipitation data (lazily; computed at read time on the
        selected chunks only).

        Parameters
        ----------
        mean: np.array
            The mean values (per pixel)
        std: np.array
            The standard deviations (per pixel)
        """
        mean = self._as_spatial_da(mean)
        std = self._as_spatial_da(std)
        precip = self.data[self.precip_var]
        self.data[self.precip_var] = ((precip - mean) / std).astype('float32')
        self._transform_tag += '_std'
        self._drop_preloaded()

    def normalize(self, q99):
        """
        Normalize the precipitation data (lazily; computed at read time on the
        selected chunks only).

        Parameters
        ----------
        q99: np.array
            The 99th quantile (per pixel)
        """
        q99 = self._as_spatial_da(q99)
        precip = self.data[self.precip_var]
        # Precipitation (raw or log1p-transformed) is non-negative, so the lower
        # bound of the normalization is 0.
        self.data[self.precip_var] = (precip / q99).astype('float32')
        self._transform_tag += '_norm'
        self._drop_preloaded()

    def log_transform(self):
        """
        Log-transform the precipitation data (lazily; computed at read time on
        the selected chunks only).
        """
        precip = self.data[self.precip_var]
        self.data[self.precip_var] = np.log1p(precip).astype('float32')
        self._transform_tag += '_log'
        self._drop_preloaded()

    def compute_mean_and_std_per_pixel(self):
        """
        Compute the mean and standard deviation of the precipitation data for each pixel.

        Returns
        -------
        np.array, np.array
            The mean and standard deviation of the precipitation data
        """
        hash_tag = self._compute_cache_hash('meanstd')
        filename = f"precip_{self.dataset_name.lower()}_meanstd_{hash_tag}.npz"
        tmp_filename = self.tmp_dir / filename

        if tmp_filename.exists():
            logger.info("Precipitation mean/sd loaded from file %s.", tmp_filename)
            cached = np.load(tmp_filename)
            return cached['mean'], cached['std']

        precip = self.data[self.precip_var]
        n_rows = precip.sizes[self.y_axis_dim]
        n_cols = precip.sizes[self.x_axis_dim]

        # Compute by spatial blocks: each block loads the whole time series for
        # its pixels (needed for the statistics) while bounding memory.
        mean = np.zeros((n_rows, n_cols))
        std = np.zeros((n_rows, n_cols))
        for i in tqdm(np.arange(0, n_rows, self.mem_nb_pixels),
                      desc="Computing mean and standard deviation per pixel"):
            for j in np.arange(0, n_cols, self.mem_nb_pixels):
                block = self._get_spatial_block(precip, i, j)
                mean[i:i + block.shape[1], j:j + block.shape[2]] = \
                    np.nanmean(block, axis=0)
                std[i:i + block.shape[1], j:j + block.shape[2]] = \
                    np.nanstd(block, axis=0)

        if not tmp_filename.exists():
            np.savez(tmp_filename, mean=mean, std=std)

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
        hash_tag = self._compute_cache_hash(f"q{quantile:.3f}")
        filename = f"precip_{self.dataset_name.lower()}_q_{quantile:.3f}_{hash_tag}.npy"
        tmp_filename = self.tmp_dir / filename

        if tmp_filename.exists():
            logger.info("Precipitation quantile %s loaded from file %s.",
                        quantile, tmp_filename)
            return np.load(tmp_filename)

        precip = self.data[self.precip_var]
        n_rows = precip.sizes[self.y_axis_dim]
        n_cols = precip.sizes[self.x_axis_dim]

        # Compute by spatial blocks: each block loads the whole time series for
        # its pixels (needed for the quantile) while bounding memory.
        quantiles = np.zeros((n_rows, n_cols))
        for i in tqdm(np.arange(0, n_rows, self.mem_nb_pixels),
                      desc=f"Computing {quantile} quantile per pixel"):
            for j in np.arange(0, n_cols, self.mem_nb_pixels):
                block = self._get_spatial_block(precip, i, j)
                quantiles[i:i + block.shape[1], j:j + block.shape[2]] = \
                    np.nanquantile(block, quantile, axis=0)

        if not tmp_filename.exists():
            np.save(tmp_filename, quantiles)

        return quantiles

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
        if self.full_grid_data is not None:
            return self.full_grid_data[self.precip_var].sel(
                time=slice(t_start, t_end),
                x=slice(x_start, x_end),
                y=slice(y_start, y_end)
            ).to_numpy()

        if self.cid_time_series is not None and cid is not None:
            try:
                ts = self.cid_time_series.sel(
                    time=slice(t_start, t_end),
                    cid=cid
                ).to_numpy()
            except KeyError as e:
                logger.error("Error with CID %s and time %s to %s", cid, t_start, t_end)
                logger.error("File: precip_%s_all_cids_[hash].nc", self.dataset_name.lower())
                logger.error("%s", e)

            # If the time series is 1D, add 2 dimensions
            if len(ts.shape) == 1:
                ts = np.expand_dims(ts, axis=(1, 2))

            return ts

        return self.data[self.precip_var].sel(
            time=slice(t_start, t_end),
            x=slice(x_start, x_end),
            y=slice(y_start, y_end)
        ).to_numpy()

    def _get_spatial_block(self, precip, i, j):
        """
        Load the full time series for the spatial block starting at (i, j) as a
        (time, y, x) numpy array.
        """
        return precip.isel({
            self.y_axis_dim: slice(i, i + self.mem_nb_pixels),
            self.x_axis_dim: slice(j, j + self.mem_nb_pixels),
        }).compute().values

    def _as_spatial_da(self, values):
        """Wrap a per-pixel (y, x) array so it broadcasts against the data."""
        if isinstance(values, xr.DataArray):
            return values
        return xr.DataArray(
            np.asarray(values),
            coords={self.y_axis_dim: self.data[self.y_axis_dim],
                    self.x_axis_dim: self.data[self.x_axis_dim]},
            dims=(self.y_axis_dim, self.x_axis_dim))

    def _compute_cache_hash(self, extra=''):
        """
        Hash identifying the current data selection (dataset, resolution, time
        step, period, spatial extent, applied transforms) for cache filenames.
        """
        h = hashlib.md5()
        h.update(str(self.dataset_name).encode())
        h.update(str(self.resolution).encode())
        h.update(str(self.time_step).encode())
        h.update(self._transform_tag.encode())
        h.update(np.asarray(self.data[self.time_axis_dim][0]).tobytes())
        h.update(np.asarray(self.data[self.time_axis_dim][-1]).tobytes())
        h.update(self.data[self.y_axis_dim].values.tobytes())
        h.update(self.data[self.x_axis_dim].values.tobytes())
        if isinstance(extra, str):
            extra = extra.encode()
        h.update(extra)

        return h.hexdigest()

    def _use_derived_store(self, resolution, time_step):
        """
        Switch to the derived zarr store for the given spatial resolution [km]
        and time step [h], materializing it once from the currently opened base
        store (kept in TMP_DIR and reused across runs).

        Parameters
        ----------
        resolution: int
            The target spatial resolution [km]
        time_step: int|float
            The target time step [h]
        """
        if resolution == self.resolution and time_step == self.time_step:
            return

        # The name must carry the period: the base data is year-sliced, so the
        # same resolution/time step over another period is a different store.
        t_first = pd.Timestamp(self.data[self.time_axis_dim].values[0])
        t_last = pd.Timestamp(self.data[self.time_axis_dim].values[-1])
        name = (f"precip_{self.dataset_name.lower()}"
                f"_r{resolution:g}_t{time_step:g}h"
                f"_{t_first.year}-{t_last.year}.zarr")
        derived_path = self.tmp_dir / name
        done_marker = Path(str(derived_path) + '.done')

        if not done_marker.exists():
            logger.info("Building derived precipitation store '%s'.", derived_path)
            self.resolution = resolution
            self.time_step = time_step
            derived = self._resample(self.data)
            derived = derived.chunk(
                {self.time_axis_dim: 720,
                 self.y_axis_dim: 32, self.x_axis_dim: 32})
            derived = derived.drop_encoding()
            # mode='w' overwrites leftovers of an interrupted build (no marker).
            derived.to_zarr(derived_path, mode='w', consolidated=False)
            done_marker.touch()

        self.data = xr.open_zarr(derived_path, consolidated=False)
        self.resolution = resolution
        self.time_step = time_step

    def _resample(self, data):
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            # Adapt the spatial resolution
            if self.resolution != 1:
                data = data.coarsen(
                    x=self.resolution,
                    y=self.resolution,
                    boundary='trim'
                ).mean()

            # Aggregate the precipitation at the desired time step. Resample whenever
            # the target step differs from the native step of the source (e.g. native
            # 5-min -> hourly). The hourly product (native == target == 1h) is a no-op.
            # Timestamps label the END of the accumulation interval (both in the
            # native 5-min files and in the hourly netCDF product), so the bins must
            # be right-closed and right-labelled: the step labelled T sums the native
            # steps over (T - target, T].
            if self.time_step is not None and self.time_step != self.native_time_step:
                data = data.resample(
                    time=f'{self.time_step}h',
                    closed='right',
                    label='right',
                ).sum(dim='time')

        return data

    @staticmethod
    def _remove_duplicate_timestamps(data):
        # Keep the first occurrence of each timestamp (positional selection:
        # label-based selection fails on a non-unique index).
        _, index = np.unique(data['time'], return_index=True)
        if len(index) != len(data['time']):
            data = data.isel(time=np.sort(index))

        return data

    def _fill_missing_values(self, data, start_time, end_time):
        # Create a complete time series index at the native frequency of the source
        freq_minutes = int(round(self.native_time_step * 60))
        freq = f'{freq_minutes}min'
        complete_time_index = pd.date_range(
            start=start_time, end=end_time, freq=freq)

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

    def _drop_preloaded(self):
        """
        Drop in-memory copies after a transform changed the (lazy) data: they
        are rebuilt (or reloaded from their per-transform cache) on demand.
        """
        self.cid_time_series = None
        self.full_grid_data = None
