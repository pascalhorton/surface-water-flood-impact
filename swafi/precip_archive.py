"""
Class to handle the precipitation archive data.
"""
import hashlib
import logging
import warnings
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

# CDF (percentile) transform. The per-pixel distribution is estimated on the wet
# time steps only: precipitation is zero-inflated, so a percentile taken over all
# steps would place every dry step at ~0.9 and squeeze the whole signal into the
# top tenth of the output range.
CDF_WET_THRESHOLD = 0.1  # [mm per time step] below this, a step counts as dry
CDF_NB_LEVELS = 40
# Rarest resolved wet value, as -log10 of its exceedance probability among wet
# steps (4 -> a 1-in-10'000 wet step). Also the maximum output value of the
# 'return_period' spread.
CDF_MAX_LOG_EXCEEDANCE = 4.0

# Floor applied to the per-pixel divisors of the standardize/normalize transforms
# [mm per time step]. A pixel that is almost always dry - common at sub-hourly
# resolution, where the wet fraction can fall below 1% - otherwise yields a zero
# standard deviation or a zero 99th percentile, and the division sends the whole
# time series of that pixel to inf/NaN. The value is small enough to leave any
# genuinely wet pixel untouched.
TRANSFORM_DIVISOR_FLOOR = 1e-3


def time_step_to_minutes(time_step_h):
    """
    Integer minutes for a time step given in hours.

    Used for the pandas frequency strings and the derived-store identity, so
    that a sub-hourly step is exact regardless of the float the caller passes
    (e.g. both 5/60 and 0.0833 map to 5). The step must be a whole number of
    minutes.

    Parameters
    ----------
    time_step_h: float
        The time step [h].

    Returns
    -------
    int
        The time step [min].
    """
    minutes = time_step_h * 60
    minutes_rounded = int(round(minutes))
    # Tolerant to one second, so a step meant as a whole number of minutes but
    # typed as a rounded decimal of an hour (0.0833 for 5/60) is accepted.
    assert abs(minutes - minutes_rounded) < 1 / 60, \
        (f"The time step ({time_step_h} h) must be a whole number of minutes "
         f"(e.g. 0.0833 = 5/60 for 5 min).")
    assert minutes_rounded > 0, "The time step must be > 0."
    return minutes_rounded


def get_cdf_levels(spread, nb_levels=CDF_NB_LEVELS):
    """
    Percentile levels of the CDF transform, and the output step between two
    consecutive levels.

    The levels are spaced so that the transform output is simply the number of
    levels the value exceeds, times the step: the output scale is built into the
    level grid rather than applied afterwards.

    Parameters
    ----------
    spread: str
        How the percentiles are spread over the output range:
        - 'return_period': the output is -log10 of the exceedance probability
          among wet steps, i.e. the log10 of the return period expressed in wet
          steps, in [0, CDF_MAX_LOG_EXCEEDANCE]. The upper tail (where damaging
          events live) gets most of the range instead of being packed against 1.
        - 'none': the output is the percentile itself, in [0, 1]. Uniformly
          distributed over the wet steps, but the extremes are compressed.
    nb_levels: int
        The number of levels of the CDF table.

    Returns
    -------
    np.array, float
        The percentile levels (increasing, starting at 0), and the output step.
    """
    if spread == 'return_period':
        step = CDF_MAX_LOG_EXCEEDANCE / nb_levels
        exponents = np.arange(nb_levels) * step
        return 1.0 - np.power(10.0, -exponents), step
    if spread == 'none':
        step = 1.0 / nb_levels
        return np.arange(nb_levels) * step, step

    raise ValueError(f"Unknown CDF spread: {spread}. "
                     f"Options are: 'return_period', 'none'")


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
        Load the (already trimmed) dataset into memory, once.

        Not an optional optimisation: it is effectively required for any window
        larger than one pixel.
        Repeat calls are a no-op. The generators are built per split and each
        one asks for the preload, so without this guard the whole domain would
        be decompressed two or three times over, holding two copies while it
        did.
        """
        if self.full_grid_data is not None:
            logger.debug("Full precipitation grid already in memory.")
            return

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
        selected chunks only). Idempotent: the train/valid/test data generators
        share the same precipitation object and each request the transform.

        Parameters
        ----------
        mean: np.array
            The mean values (per pixel)
        std: np.array
            The standard deviations (per pixel)
        """
        if '_std' in self._transform_tag:
            logger.debug("Precipitation already standardized; skipping.")
            return

        std = self._floor_divisor(std, 'standard deviation')
        mean = self._as_spatial_da(mean)
        std = self._as_spatial_da(std)
        precip = self.data[self.precip_var]
        self.data[self.precip_var] = ((precip - mean) / std).astype('float32')
        self._transform_tag += '_std'
        self._drop_preloaded()

    def normalize(self, q99):
        """
        Normalize the precipitation data (lazily; computed at read time on the
        selected chunks only). Idempotent: the train/valid/test data generators
        share the same precipitation object and each request the transform.

        Parameters
        ----------
        q99: np.array
            The 99th quantile (per pixel)
        """
        if '_norm' in self._transform_tag:
            logger.debug("Precipitation already normalized; skipping.")
            return

        q99 = self._floor_divisor(q99, '99th quantile')
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
        the selected chunks only). Idempotent: the train/valid/test data
        generators share the same precipitation object and each request the
        transform.
        """
        if '_log' in self._transform_tag:
            logger.debug("Precipitation already log-transformed; skipping.")
            return

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

    def compute_cdf_table_per_pixel(self, levels, wet_threshold=CDF_WET_THRESHOLD):
        """
        Compute, for each pixel, the precipitation value at each percentile level
        of its wet-step distribution (the value grid inverted by cdf_transform).

        Parameters
        ----------
        levels: np.array
            The percentile levels, from get_cdf_levels().
        wet_threshold: float
            Time steps at or below this value are excluded from the distribution
            (and map to 0 by the transform).

        Returns
        -------
        np.array
            The value grid, (nb_levels, nb_rows, nb_cols). Pixels with no wet
            step hold +inf, so that the transform maps them to 0.
        """
        levels = np.asarray(levels, dtype='float64')
        hash_tag = self._compute_cache_hash(
            b'cdftable' + levels.tobytes() + str(wet_threshold).encode())
        filename = f"precip_{self.dataset_name.lower()}_cdf_{hash_tag}.npy"
        tmp_filename = self.tmp_dir / filename

        if tmp_filename.exists():
            logger.info("Precipitation CDF table loaded from file %s.",
                        tmp_filename)
            return np.load(tmp_filename)

        precip = self.data[self.precip_var]
        n_rows = precip.sizes[self.y_axis_dim]
        n_cols = precip.sizes[self.x_axis_dim]

        # Compute by spatial blocks: each block loads the whole time series for
        # its pixels (needed for the quantiles) while bounding memory.
        table = np.zeros((len(levels), n_rows, n_cols))
        for i in tqdm(np.arange(0, n_rows, self.mem_nb_pixels),
                      desc="Computing the CDF table per pixel"):
            for j in np.arange(0, n_cols, self.mem_nb_pixels):
                block = self._get_spatial_block(precip, i, j)
                # Masked in place: a copy would double the memory of a block
                # holding the whole time series of its pixels.
                block[block <= wet_threshold] = np.nan
                with warnings.catch_warnings():
                    # Pixels without any wet step: handled right below.
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    table[:, i:i + block.shape[1], j:j + block.shape[2]] = \
                        np.nanquantile(block, levels, axis=0)

        nb_dry_pixels = int(np.isnan(table[0]).sum())
        if nb_dry_pixels:
            logger.warning("%d pixels have no wet time step (above %s); they "
                           "are mapped to 0.", nb_dry_pixels, wet_threshold)
        table = np.where(np.isfinite(table), table, np.inf)

        if not tmp_filename.exists():
            np.save(tmp_filename, table)

        return table

    def cdf_transform(self, levels, table, step):
        """
        Replace each value by its rank in the wet-step distribution of its own
        pixel (lazily; computed at read time on the selected chunks only).

        This is the transform the ``*_q`` event features use, applied to the
        precipitation series itself: it makes intensities comparable between
        pixels with different climatologies. Dry steps map to 0. Idempotent: the
        train/valid/test data generators share the same precipitation object and
        each request the transform.

        Parameters
        ----------
        levels: np.array
            The percentile levels, from get_cdf_levels().
        table: np.array
            The per-pixel value grid, from compute_cdf_table_per_pixel().
        step: float
            The output step between two consecutive levels, from
            get_cdf_levels().
        """
        if '_cdf' in self._transform_tag:
            logger.debug("Precipitation already CDF-transformed; skipping.")
            return

        precip = self.data[self.precip_var]

        # The output is the number of levels the value exceeds, times the step.
        # Accumulated one level at a time so that no (time, level, y, x) array is
        # ever materialized.
        counts = None
        for values in table:
            # Cast before accumulating: '+' on booleans is a logical or.
            exceeds = (precip > self._as_spatial_da(values)).astype('float32')
            counts = exceeds if counts is None else counts + exceeds

        self.data[self.precip_var] = (counts * step).astype('float32')
        # The tag identifies the level grid, not just its size: two spreads can
        # resolve the same number of levels ('return_period' and 'none' both use
        # CDF_NB_LEVELS) while being different transforms. Tagging them alike
        # would let the second one read the first one's cached series back.
        grid_id = hashlib.md5(
            np.asarray(levels, dtype='float64').tobytes()
            + str(step).encode()).hexdigest()[:8]
        self._transform_tag += f'_cdf{len(levels)}-{grid_id}'
        self._drop_preloaded()

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

    @staticmethod
    def _floor_divisor(values, label):
        """
        Clip a per-pixel divisor away from zero before it is used to rescale the
        precipitation. Pixels that are dry over the whole record yield a divisor
        of 0 (or NaN, when the statistic was computed on an empty selection), and
        dividing by it turns the entire time series of that pixel into inf/NaN,
        which then propagates silently through the network and shows up only as a
        NaN loss many epochs later.

        Parameters
        ----------
        values: np.array
            The per-pixel divisor (standard deviation or quantile).
        label: str
            Name of the statistic, for the warning message.

        Returns
        -------
        np.array
            The divisor, with every entry at or above TRANSFORM_DIVISOR_FLOOR.
        """
        values = np.asarray(values, dtype='float64')
        degenerate = ~np.isfinite(values) | (values < TRANSFORM_DIVISOR_FLOOR)
        nb_degenerate = int(degenerate.sum())

        if nb_degenerate:
            logger.warning(
                "%d of %d pixels have a %s below %s (dry over the whole record); "
                "clipping to that floor so the transform stays finite.",
                nb_degenerate, values.size, label, TRANSFORM_DIVISOR_FLOOR)

        values = np.where(degenerate, TRANSFORM_DIVISOR_FLOOR, values)

        return values

    def match_stats_to_grid(self, values, label='statistic'):
        """
        Align a per-pixel statistic to the current precipitation grid.

        Statistics saved by compute_precipitation_statistics.py cover the domain
        of the store they were computed on, which is not necessarily the one
        being read now (the zarr store is cropped to the CID domain). When the
        statistic carries coordinates, the pixels of the current grid are picked
        out of it; otherwise its shape must already match.

        Parameters
        ----------
        values: xr.DataArray|np.array
            The per-pixel statistic.
        label: str
            Name of the statistic, for the error message.

        Returns
        -------
        np.array
            The statistic on the current grid.

        Raises
        ------
        ValueError
            If the statistic cannot be matched to the current grid.
        """
        y_axis = self.data[self.y_axis_dim]
        x_axis = self.data[self.x_axis_dim]
        grid_shape = (y_axis.size, x_axis.size)

        if isinstance(values, xr.DataArray):
            source_shape = values.shape
            if source_shape == grid_shape:
                return values.to_numpy()
            if self.y_axis_dim not in values.coords or \
                    self.x_axis_dim not in values.coords:
                raise ValueError(
                    f"The {label} has shape {values.shape} but the precipitation "
                    f"grid is {grid_shape}, and it carries no "
                    f"'{self.y_axis_dim}'/'{self.x_axis_dim}' coordinates to "
                    f"select from.")
            try:
                values = values.sel({self.y_axis_dim: y_axis,
                                     self.x_axis_dim: x_axis})
            except KeyError as e:
                raise ValueError(
                    f"The {label} does not cover the precipitation grid: some "
                    f"cells of the current domain are missing from it "
                    f"({e}).") from e
            logger.info(
                "Selected the %s of the %s stats grid on the current "
                "precipitation grid (%s).", label, source_shape, grid_shape)

            return values.to_numpy()

        values = np.asarray(values)
        if values.shape != grid_shape:
            raise ValueError(
                f"The {label} has shape {values.shape}, which does not match the "
                f"precipitation grid {grid_shape}. Recompute it for this domain "
                f"(scripts/data_preparation/compute_precipitation_statistics.py) "
                f"or pass it as a DataArray with "
                f"'{self.y_axis_dim}'/'{self.x_axis_dim}' coordinates.")

        return values

    def _as_spatial_da(self, values):
        """Wrap a per-pixel (y, x) array so it broadcasts against the data."""
        if isinstance(values, xr.DataArray):
            return values
        values = self.match_stats_to_grid(values, 'per-pixel statistic')
        return xr.DataArray(
            values,
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
        # Compared in minutes so that a native step passed as a slightly
        # imprecise float (e.g. 0.0833 for 5/60 h) is recognised as native and
        # does not trigger a needless rebuild.
        target_minutes = time_step_to_minutes(time_step)
        current_minutes = (time_step_to_minutes(self.time_step)
                           if self.time_step is not None else None)
        if resolution == self.resolution and target_minutes == current_minutes:
            return

        # The name must carry the period: the base data is year-sliced, so the
        # same resolution/time step over another period is a different store.
        t_first = pd.Timestamp(self.data[self.time_axis_dim].values[0])
        t_last = pd.Timestamp(self.data[self.time_axis_dim].values[-1])
        name = (f"precip_{self.dataset_name.lower()}"
                f"_r{resolution:g}_t{target_minutes}min"
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
            if self.time_step is not None and \
                    time_step_to_minutes(self.time_step) != \
                    time_step_to_minutes(self.native_time_step):
                data = data.resample(
                    time=f'{time_step_to_minutes(self.time_step)}min',
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
