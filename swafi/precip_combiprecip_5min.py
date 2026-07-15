"""
Class to handle the 5-minute precipitation data from CombiPrecip (CPCH).
"""

import concurrent.futures
import io
import logging
import re
import zipfile
from pathlib import Path

import dask
import dask.array as da
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .config import Config
from .precip_archive import PrecipitationArchive

config = Config()

logger = logging.getLogger(__name__)

# Canonical CombiPrecip grid (EPSG:2056 / CH1903+ / LV95), 1 km cells.
# Full grid covers x 2255000->2965000 and y 840000->1480000.
# Pixel centres are offset by +500 m from the lower-left corner.
GRID_X_SIZE = 710
GRID_Y_SIZE = 640
GRID_X0 = 2255500.0  # centre of the first (westmost) column
GRID_Y0 = 1479500.0  # centre of the first (northmost) row
GRID_RESOLUTION = 1000.0

# Number of 5-min steps in a day and the native time step (in hours).
STEPS_PER_DAY = 288
NATIVE_TIME_STEP = 5 / 60  # hours
RATE_TO_STEP = 5.0 / 60.0  # mm/h rate -> mm accumulated over the 5-min step

# 5-min accumulation files inside the daily zips, e.g. CPC2400100054_00005.001.h5.
# The name encodes the timestamp as <YY><DOY><HH><MM> followed by a quality digit;
# the version suffix varies with the product version (.000 / .801 / .001). The
# timestamp labels the END of the 5-min accumulation interval (same convention as
# the hourly netCDF product). The HDF5 'what' date/time attributes must NOT be
# used: they are rounded up (to the hour before 2024, to 10 min in 2024).
FILENAME_PATTERN = re.compile(
    r'CPC(\d{2})(\d{3})(\d{2})(\d{2})\d_00005\.\d{3}\.h5$')


def _read_day_zip(zip_path, date):
    """
    Read all 5-min HDF5 files contained in a daily CPCH zip into a single array.

    Parameters
    ----------
    zip_path: str|Path
        The path to the daily zip file (e.g. CPCHhdf524001.zip).
    date: pd.Timestamp
        The date (00:00) of the day covered by the zip.

    Returns
    -------
    np.ndarray
        A (STEPS_PER_DAY, GRID_Y_SIZE, GRID_X_SIZE) float32 array. Time steps with
        no corresponding file are filled with NaN.
    """
    expected = pd.date_range(date, periods=STEPS_PER_DAY, freq='5min')
    pos = {t: i for i, t in enumerate(expected)}
    out = np.full((STEPS_PER_DAY, GRID_Y_SIZE, GRID_X_SIZE), np.nan, dtype='float32')

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # Keep only the 5-min accumulation files, ignore 60-min, daily, etc.
            match = FILENAME_PATTERN.search(member)
            if not match:
                continue

            # Parse the timestamp from the file name (the HDF5 attributes are
            # rounded and unusable, see FILENAME_PATTERN).
            yy, doy, hh, mm = (int(g) for g in match.groups())
            ts = pd.Timestamp(year=2000 + yy, month=1, day=1) + pd.Timedelta(
                days=doy - 1, hours=hh, minutes=mm)

            idx = pos.get(ts)
            if idx is None:
                continue

            with zf.open(member) as fh:
                buf = io.BytesIO(fh.read())

            with h5py.File(buf, 'r') as h5:
                raw = h5['dataset1/data1/data'][:]

            # undetect is encoded as +inf (no rain) -> 0; nodata stays NaN.
            raw = np.where(np.isposinf(raw), 0.0, raw)
            out[idx] = (raw * RATE_TO_STEP).astype('float32')

    return out


def _init_zarr_template(zarr_path, time_coord, y_coord, x_coord, chunk_size):
    """
    Write the metadata of an empty zarr store (no chunk data): unwritten chunks
    read back as NaN. The actual data is filled per day with _write_day_to_zarr.
    """
    template = xr.Dataset(
        {'precip': (
            ('time', 'y', 'x'),
            da.full((len(time_coord), len(y_coord), len(x_coord)),
                    np.nan, dtype='float32',
                    chunks=(STEPS_PER_DAY, chunk_size, chunk_size))
        )},
        coords={'time': time_coord, 'y': y_coord, 'x': x_coord}
    )
    # consolidated=False: consolidated metadata is not part of the zarr v3 spec
    # and only triggers warnings; the store holds a single array anyway.
    template.to_zarr(zarr_path, compute=False, consolidated=False,
                     encoding={'precip': {'_FillValue': np.float32(np.nan)}})


def _write_day_to_zarr(zarr_path, zip_path, date, day_idx, y_start, y_end,
                       x_start, x_end, marker_path):
    """
    Read one daily CPCH zip and write it into its time region of the zarr store.

    Parameters
    ----------
    zarr_path: str
        The path to the zarr store.
    zip_path: str
        The path to the daily zip file.
    date: pd.Timestamp
        The date (00:00) of the day covered by the zip.
    day_idx: int
        The index of the day in the store's calendar (0 = first day).
    y_start, y_end, x_start, x_end: int
        The full-grid index bounds of the spatial crop stored in the zarr.
    marker_path: str
        The path of the marker file to create once the day is written.
    """
    arr = _read_day_zip(zip_path, date)
    arr = arr[:, y_start:y_end, x_start:x_end]

    # No coordinate variables: only the 'precip' region is written.
    day = xr.Dataset({'precip': (('time', 'y', 'x'), arr)})
    t0 = day_idx * STEPS_PER_DAY
    day.to_zarr(zarr_path, region={'time': slice(t0, t0 + STEPS_PER_DAY)},
                consolidated=False)

    Path(marker_path).touch()

    return True


class CombiPrecip5min(PrecipitationArchive):
    def __init__(self, year_start=None, year_end=None, cid_file=None):
        """
        The Precipitation class for the 5-minute CombiPrecip (CPCH) data. Reads the
        zipped ODIM_H5 HDF5 files organised as <YEAR>/<DOY>/CPCH*.zip.

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
        self.dataset_name = "CombiPrecip5min"
        self.native_time_step = NATIVE_TIME_STEP

    def open_files(self, data_path=None, resolution=1, time_step=None):
        """
        Open the 5-minute precipitation data from the given path.

        Parameters
        ----------
        data_path: str|None
            The path to the data files (root of the <YEAR>/<DOY>/*.zip tree)
        resolution: int
            The spatial resolution [km] of the precipitation data (default: 1)
        time_step: int|None
            The target time step [h]. If None (default), the native 5-min resolution
            is kept. Aggregation to a coarser step happens during pickle generation.
        """
        if data_path:
            self.data_path = data_path
        if not self.data_path:
            self.data_path = config.get('DIR_PRECIP_5MIN')
        if not self.data_path:
            raise FileNotFoundError("The data path was not provided.")
        self.resolution = resolution
        self.time_step = time_step

        day_list = self._list_daily_zips()
        if not day_list:
            raise FileNotFoundError(
                f"No CPCH zip files found in {self.data_path} for "
                f"{self.year_start}-{self.year_end}.")

        delayed_arrays = []
        time_chunks = []
        for date, zip_path in day_list:
            arr = da.from_delayed(
                dask.delayed(_read_day_zip)(str(zip_path), date),
                shape=(STEPS_PER_DAY, GRID_Y_SIZE, GRID_X_SIZE),
                dtype='float32'
            )
            delayed_arrays.append(arr)
            time_chunks.append(pd.date_range(date, periods=STEPS_PER_DAY, freq='5min'))

        precip = da.concatenate(delayed_arrays, axis=0)
        time_coord = pd.DatetimeIndex(np.concatenate(
            [t.to_numpy() for t in time_chunks]))

        x_axis = GRID_X0 + np.arange(GRID_X_SIZE) * GRID_RESOLUTION
        y_axis = GRID_Y0 - np.arange(GRID_Y_SIZE) * GRID_RESOLUTION

        self.data = xr.Dataset(
            {'precip': (('time', 'y', 'x'), precip)},
            coords={'time': time_coord, 'y': y_axis, 'x': x_axis}
        )

        # Select the data for the given years (mirrors CombiPrecip.open_files)
        if self.year_start:
            self.data = self.data.sel(time=slice(f'{self.year_start}-01-01', None))
        if self.year_end:
            self.data = self.data.sel(time=slice(None, f'{self.year_end}-12-31'))

    def open_zarr(self, zarr_path=None):
        """
        Open the 5-minute precipitation data from a zarr store built with
        build_zarr_store(). The data is opened lazily; only the chunks actually
        selected are read from disk.

        Parameters
        ----------
        zarr_path: str|Path|None
            The path to the zarr store. Defaults to the PATH_PRECIP_5MIN_ZARR
            config entry.
        """
        if not zarr_path:
            zarr_path = config.get('PATH_PRECIP_5MIN_ZARR')
        if not zarr_path or not Path(zarr_path).exists():
            raise FileNotFoundError(
                f"The zarr store '{zarr_path}' does not exist. Build it first with "
                f"scripts/data_preparation/build_precip_5min_zarr.py.")

        self.data = xr.open_zarr(zarr_path, consolidated=False)
        self.resolution = 1
        self.time_step = NATIVE_TIME_STEP

        # Select the data for the given years (mirrors open_files)
        if self.year_start:
            self.data = self.data.sel(time=slice(f'{self.year_start}-01-01', None))
        if self.year_end:
            self.data = self.data.sel(time=slice(None, f'{self.year_end}-12-31'))

    def build_zarr_store(self, zarr_path, data_path=None, n_workers=4,
                         chunk_size=64, margin=5000):
        """
        Convert the daily CPCH zips into a compressed, spatially-chunked zarr store
        (single pass over the source data). The store is cropped to the CID domain
        bounding box (plus a margin) and chunked one day x chunk_size x chunk_size,
        so that the extraction can later read small spatial tiles over the full
        period without materialising the full grid.

        The build is resumable: days already written (tracked with marker files in
        '<zarr_path>.done/') are skipped.

        Parameters
        ----------
        zarr_path: str|Path
            The path of the zarr store to create/complete.
        data_path: str|None
            The path to the source zips (root of the <YEAR>/<DOY>/*.zip tree).
            Defaults to the DIR_PRECIP_5MIN config entry.
        n_workers: int
            The number of parallel processes writing days to the store.
        chunk_size: int
            The spatial chunk size [cells] of the store (default: 64).
        margin: float
            The margin [m] added around the CID domain extent (default: 5000).
        """
        if data_path:
            self.data_path = data_path
        if not self.data_path:
            self.data_path = config.get('DIR_PRECIP_5MIN')
        if not self.data_path:
            raise FileNotFoundError("The data path was not provided.")

        day_list = self._list_daily_zips()
        if not day_list:
            raise FileNotFoundError(
                f"No CPCH zip files found in {self.data_path} for "
                f"{self.year_start}-{self.year_end}.")

        # Full calendar of the store; days without a zip stay at the fill value
        # (NaN). Day i of the calendar maps to time steps [i*288, (i+1)*288).
        t0 = pd.Timestamp(year=self.year_start, month=1, day=1)
        t_end = pd.Timestamp(year=self.year_end, month=12, day=31)
        n_days = (t_end - t0).days + 1
        time_coord = pd.date_range(t0, periods=n_days * STEPS_PER_DAY, freq='5min')

        # Crop the full grid to the CID domain bounding box (plus margin)
        x_axis = GRID_X0 + np.arange(GRID_X_SIZE) * GRID_RESOLUTION
        y_axis = GRID_Y0 - np.arange(GRID_Y_SIZE) * GRID_RESOLUTION
        extent = self.domain.cids['extent']
        x_idx = np.where((x_axis >= extent.left - margin) &
                         (x_axis <= extent.right + margin))[0]
        y_idx = np.where((y_axis >= extent.bottom - margin) &
                         (y_axis <= extent.top + margin))[0]
        x_start, x_end = int(x_idx[0]), int(x_idx[-1]) + 1
        y_start, y_end = int(y_idx[0]), int(y_idx[-1]) + 1

        zarr_path = Path(zarr_path)
        done_dir = Path(str(zarr_path) + '.done')
        done_dir.mkdir(exist_ok=True)

        # The store is initialized iff its metadata file exists: a bare directory
        # (created manually or by an aborted run) must still get the template.
        if (zarr_path / 'zarr.json').exists():
            existing = xr.open_zarr(zarr_path, consolidated=False)
            store_start = pd.Timestamp(existing['time'].values[0])
            store_steps = existing.sizes['time']
            existing.close()
            if store_steps != len(time_coord) or store_start != time_coord[0]:
                raise ValueError(
                    f"The existing zarr store '{zarr_path}' covers a different "
                    f"calendar (starts {store_start}, {store_steps} steps) than "
                    f"requested ({time_coord[0]}, {len(time_coord)} steps): the "
                    f"day-to-region mapping would corrupt it. Delete the store "
                    f"and '{done_dir}' or adjust year_start/year_end.")
        else:
            stale_markers = list(done_dir.iterdir())
            if stale_markers:
                logger.warning("Removing %d stale day markers from '%s' "
                               "(no initialized store found).",
                               len(stale_markers), done_dir)
                for marker in stale_markers:
                    marker.unlink()
            _init_zarr_template(zarr_path, time_coord, y_axis[y_start:y_end],
                                x_axis[x_start:x_end], chunk_size)
            logger.info("Initialized zarr store '%s' (%d days, %d x %d cells).",
                        zarr_path, n_days, y_end - y_start, x_end - x_start)

        todo = [(date, zip_path) for date, zip_path in day_list
                if not (done_dir / date.strftime('%Y-%m-%d')).exists()]
        logger.info("%d days to write (%d already done).",
                    len(todo), len(day_list) - len(todo))

        # One day maps to exactly one time chunk, so parallel processes never
        # write to the same chunk.
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _write_day_to_zarr, str(zarr_path), str(zip_path), date,
                    (date - t0).days, y_start, y_end, x_start, x_end,
                    str(done_dir / date.strftime('%Y-%m-%d')))
                for date, zip_path in todo
            ]
            for f in tqdm(concurrent.futures.as_completed(futures),
                          total=len(futures), desc="Writing days to zarr"):
                f.result()

        n_missing = n_days - len(day_list)
        if n_missing > 0:
            logger.warning("%d calendar days have no source zip and stay NaN.",
                           n_missing)
        logger.info("Zarr store '%s' complete.", zarr_path)

    def prepare_data(self, data_path=None, resolution=1, time_step=1):
        """
        Load the precipitation data and generate the monthly pickle files.

        Parameters
        ----------
        data_path: str|None
            The path to the data files
        resolution: int
            The spatial resolution [km] of the precipitation data (default: 1)
        time_step: int
            The target time step [h] of the precipitation data (default: 1). The
            native 5-min data is aggregated (summed) to this step.
        """
        self.open_files(data_path, resolution, time_step)
        self._generate_pickle_files()

    def _list_daily_zips(self):
        """
        List the daily CPCH zip files for the requested year range.

        Returns
        -------
        list of (pd.Timestamp, Path)
            The (date, zip_path) pairs, sorted by date.
        """
        base = Path(self.data_path)
        day_list = []
        for year in range(self.year_start, self.year_end + 1):
            year_dir = base / str(year)
            if not year_dir.is_dir():
                continue
            for doy_dir in sorted(year_dir.iterdir()):
                if not doy_dir.is_dir():
                    continue
                zips = sorted(doy_dir.glob('CPCH*.zip'))
                if not zips:
                    continue
                # The DOY folder is named <YY><DOY>, e.g. '24001'. Some folders
                # contain stray zips from another day (e.g. 2023/23001 also holds
                # CPCHhdf513001.zip), so pick the one matching the folder name.
                if len(zips) > 1:
                    zips = [z for z in zips
                            if z.name == f'CPCHhdf5{doy_dir.name}.zip']
                    if len(zips) != 1:
                        raise ValueError(
                            f"Cannot identify the CPCH zip for {doy_dir}")
                doy = int(doy_dir.name[2:])
                date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(
                    days=doy - 1)
                day_list.append((date, zips[0]))

        day_list.sort(key=lambda pair: pair[0])

        return day_list
