"""
Class to handle the precipitation data from CombiPrecip.
"""

import logging
from glob import glob
from pathlib import Path

import pandas as pd
import xarray as xr
from tqdm import tqdm

from .config import Config
from .precip_archive import PrecipitationArchive
from .utils.zarr_store import (ensure_zarr_store, finalize_zarr_store,
                               write_time_region)

config = Config()

logger = logging.getLogger(__name__)


class CombiPrecip(PrecipitationArchive):
    def __init__(self, year_start=None, year_end=None, cid_file=None):
        """
        The Precipitation class for CombiPrecip data. Must be netCDF files.

        Parameters
        ----------
        year_start: int
            The start year of the data
        year_end: int
            The end year of the data
        cid_file: str|None
            The path to the CID file
        """
        super().__init__(year_start, year_end, cid_file)
        self.dataset_name = "CombiPrecip"

    def open_files(self, data_path=None, resolution=1, time_step=1):
        """
        Open the precipitation data from the given path.

        Parameters
        ----------
        data_path: str|None
            The path to the data files
        resolution: int
            The resolution [km] of the precipitation data (default: 1)
        time_step: int
            The time step [h] of the precipitation data (default: 1)
        """
        if data_path:
            self.data_path = data_path
        if not self.data_path:
            self.data_path = config.get('DIR_PRECIP_HOURLY')
        if not self.data_path:
            raise FileNotFoundError("The data path was not provided.")
        self.resolution = resolution
        self.time_step = time_step

        files = sorted(glob(f"{self.data_path}/*.nc"))
        self._check_files(files)
        self.data = xr.open_mfdataset(
            files,
            parallel=False,
            chunks={'time': 1000}
        )
        self.data = self.data.rename_vars({'CPC': 'precip'})
        self.data = self.data.rename({'REFERENCE_TS': 'time'})

        # Select the data for the given years
        if self.year_start:
            self.data = self.data.sel(time=slice(f'{self.year_start}-01-01', None))
        if self.year_end:
            self.data = self.data.sel(time=slice(None, f'{self.year_end}-12-31'))

    def open_zarr(self, zarr_path=None):
        """
        Open the hourly precipitation data from a zarr store built with
        build_zarr_store(). The data is opened lazily; only the chunks actually
        selected are read from disk.

        Parameters
        ----------
        zarr_path: str|Path|None
            The path to the zarr store. Defaults to the PATH_PRECIP_HOURLY_ZARR
            config entry.
        """
        if not zarr_path:
            zarr_path = config.get('PATH_PRECIP_HOURLY_ZARR', do_raise=False)
        if not zarr_path or not (Path(zarr_path) / 'zarr.json').exists():
            where = f"'{zarr_path}'" if zarr_path else "(PATH_PRECIP_HOURLY_ZARR not set)"
            raise FileNotFoundError(
                f"The hourly zarr store {where} does not exist. Build it first "
                f"with scripts/data_preparation/build_precip_hourly_zarr.py "
                f"(config key PATH_PRECIP_HOURLY_ZARR).")

        super().open_zarr(zarr_path)

    def prepare_data(self, data_path=None, resolution=1, time_step=1):
        """
        Open the precipitation data from the base zarr store (see
        build_zarr_store) and switch to the derived store for the requested
        resolution/time step (materialized once, then reused).

        Parameters
        ----------
        data_path: str|None
            The path to the base zarr store (defaults to the
            PATH_PRECIP_HOURLY_ZARR config entry)
        resolution: int
            The resolution [km] of the precipitation data (default: 1)
        time_step: int
            The time step [h] of the precipitation data (default: 1)
        """
        self.open_zarr(data_path)
        self._use_derived_store(resolution, time_step)

    def build_zarr_store(self, zarr_path=None, data_path=None, margin=5000,
                         time_chunk=720, spatial_chunk=32):
        """
        Convert the hourly netCDF files into a compressed, spatially-chunked
        zarr store (single pass over the source data). The store is cropped to
        the CID domain bounding box (plus a margin) and cleaned exactly like the
        former monthly pickles (duplicate timestamps removed, missing steps
        linearly interpolated, remaining NaN set to 0).

        The build is resumable: months already written (tracked with marker
        files in '<zarr_path>.done/') are skipped. The marker directory is
        removed once the build completes; its absence marks a completed store
        and makes rerunning the build a no-op.

        Parameters
        ----------
        zarr_path: str|Path|None
            The path of the zarr store to create/complete. Defaults to the
            PATH_PRECIP_HOURLY_ZARR config entry.
        data_path: str|None
            The path to the source netCDF files. Defaults to the DIR_PRECIP_HOURLY
            config entry.
        margin: float
            The margin [m] added around the CID domain extent (default: 5000).
        time_chunk: int
            The time chunk size [steps] of the store (default: 720). Small
            spatial chunks with longer time chunks make both patch reads and
            full-time-series-per-pixel reads (normalization statistics) cheap.
        spatial_chunk: int
            The spatial chunk size [cells] of the store (default: 32).
        """
        if not zarr_path:
            zarr_path = config.get('PATH_PRECIP_HOURLY_ZARR', do_raise=False)
        if not zarr_path:
            raise ValueError("No zarr store path given and "
                             "PATH_PRECIP_HOURLY_ZARR is not set.")

        self.open_files(data_path)

        # Full calendar of the store (timestamps label the END of the hourly
        # accumulation interval: 00:00 of Jan 1 to 23:00 of Dec 31).
        t0 = pd.Timestamp(f'{self.year_start}-01-01 00:00')
        t_end = pd.Timestamp(f'{self.year_end}-12-31 23:00')
        time_coord = pd.date_range(t0, t_end, freq='h')

        # Crop the grid to the CID domain bounding box (plus margin)
        x_axis = self.data[self.x_axis_dim].values
        y_axis = self.data[self.y_axis_dim].values
        extent = self.domain.cids['extent']
        x_sel = x_axis[(x_axis >= extent.left - margin) &
                       (x_axis <= extent.right + margin)]
        y_sel = y_axis[(y_axis >= extent.bottom - margin) &
                       (y_axis <= extent.top + margin)]

        zarr_path = Path(zarr_path)
        done_dir = Path(str(zarr_path) + '.done')
        if ensure_zarr_store(zarr_path, time_coord, y_sel, x_sel,
                             (time_chunk, spatial_chunk, spatial_chunk),
                             done_dir):
            return

        # Written sequentially month by month (~250 MB in memory per month);
        # sequential region writes make chunk-straddling months safe.
        months = pd.date_range(t0, t_end, freq='MS')
        for month in tqdm(months, desc="Writing months to zarr"):
            marker = done_dir / month.strftime('%Y-%m')
            if marker.exists():
                continue

            m_start = month
            m_end = min(month + pd.offsets.MonthEnd(0)
                        + pd.Timedelta(hours=23), t_end)
            subset = self.data[[self.precip_var]].sel(
                time=slice(m_start, m_end))
            subset = self._remove_duplicate_timestamps(subset)
            subset = self._fill_missing_values(subset, m_start, m_end)
            subset = subset.sel({self.x_axis_dim: x_sel, self.y_axis_dim: y_sel})
            values = subset[self.precip_var].compute().values.astype('float32')

            t_offset = int((m_start - t0) / pd.Timedelta(hours=1))
            write_time_region(zarr_path, values, t_offset)
            marker.touch()

        finalize_zarr_store(zarr_path, done_dir)

    def _check_files(self, files):
        """
        The original file are named such as: CPC_00060_H_20221128000000_20221204230000.nc
        Here, we check that there is no overlapping period between files (not handled
        by xarray) using the dates in the file names.

        Parameters
        ----------
        files: list
            The list of files

        Raises
        ------
        ValueError
            If there is an overlapping period between files
        """
        for i in range(1, len(files)):
            file1 = files[i-1]
            file2 = files[i]
            date1 = file1.split('_')[-1]
            date2 = file2.split('_')[-2]
            if date1 >= date2:
                filename1 = file1.split('/')[-1]
                filename1 = filename1.split('\\')[-1]
                filename2 = file2.split('/')[-1]
                filename2 = filename2.split('\\')[-1]
                raise ValueError(f"Overlapping period between {filename1} and {filename2}")
