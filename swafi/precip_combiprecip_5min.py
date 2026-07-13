"""
Class to handle the 5-minute precipitation data from CombiPrecip (CPCH).
"""

import io
import re
import zipfile
from pathlib import Path

import dask
import dask.array as da
import h5py
import numpy as np
import pandas as pd
import xarray as xr

from .config import Config
from .precip_archive import PrecipitationArchive

config = Config()

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
