"""
Helpers to build zarr stores for gridded precipitation archives.

A store holds a single float32 'precip' variable with dims (time, y, x) on a
full-calendar time axis. It is initialized once as an empty template (unwritten
chunks read back as the NaN fill value) and then filled by time-region writes.
"""
import logging
import shutil
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def init_zarr_template(zarr_path, time_coord, y_coord, x_coord, chunks):
    """
    Write the metadata of an empty zarr store (no chunk data): unwritten chunks
    read back as NaN. The actual data is filled with write_time_region().

    Parameters
    ----------
    zarr_path: str|Path
        The path of the zarr store to create.
    time_coord: pd.DatetimeIndex
        The full time axis of the store.
    y_coord: np.ndarray
        The y coordinates (grid cell centres).
    x_coord: np.ndarray
        The x coordinates (grid cell centres).
    chunks: tuple
        The chunk sizes as (time, y, x).
    """
    template = xr.Dataset(
        {'precip': (
            ('time', 'y', 'x'),
            da.full((len(time_coord), len(y_coord), len(x_coord)),
                    np.nan, dtype='float32', chunks=chunks)
        )},
        coords={'time': time_coord, 'y': y_coord, 'x': x_coord}
    )
    # consolidated=False: consolidated metadata is not part of the zarr v3 spec
    # and only triggers warnings; the store holds a single array anyway.
    template.to_zarr(zarr_path, compute=False, consolidated=False,
                     encoding={'precip': {'_FillValue': np.float32(np.nan)}})


def ensure_zarr_store(zarr_path, time_coord, y_coord, x_coord, chunks, done_dir):
    """
    Make sure an initialized zarr store with the given calendar exists at
    zarr_path, creating the template if needed.

    The store is considered initialized iff its metadata file ('zarr.json')
    exists: a bare directory (created manually or by an aborted run) still gets
    the template. An existing store covering a different calendar raises (the
    time-region mapping of subsequent writes would corrupt it). When a fresh
    template is written, stale completion markers in done_dir are removed.

    Parameters
    ----------
    zarr_path: str|Path
        The path of the zarr store.
    time_coord: pd.DatetimeIndex
        The full time axis of the store.
    y_coord: np.ndarray
        The y coordinates.
    x_coord: np.ndarray
        The x coordinates.
    chunks: tuple
        The chunk sizes as (time, y, x).
    done_dir: str|Path
        The directory holding per-region completion markers (created if needed;
        removed by finalize_zarr_store once the build completes).

    Returns
    -------
    bool
        True if the store exists and its build already completed (no marker
        directory left, see finalize_zarr_store): nothing needs writing.
    """
    zarr_path = Path(zarr_path)
    done_dir = Path(done_dir)

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
                f"time-region mapping would corrupt it. Delete the store "
                f"and '{done_dir}' or adjust year_start/year_end.")
        if not done_dir.exists():
            logger.info("Zarr store '%s' was already completed.", zarr_path)
            return True
        return False

    done_dir.mkdir(exist_ok=True)
    stale_markers = list(done_dir.iterdir())
    if stale_markers:
        logger.warning("Removing %d stale completion markers from '%s' "
                       "(no initialized store found).",
                       len(stale_markers), done_dir)
        for marker in stale_markers:
            marker.unlink()
    init_zarr_template(zarr_path, time_coord, y_coord, x_coord, chunks)
    logger.info("Initialized zarr store '%s' (%d steps, %d x %d cells).",
                zarr_path, len(time_coord), len(y_coord), len(x_coord))
    return False


def finalize_zarr_store(zarr_path, done_dir):
    """
    Mark a build as complete by removing the marker directory: its absence
    (with the store present) is what ensure_zarr_store treats as 'completed'.

    Parameters
    ----------
    zarr_path: str|Path
        The path of the zarr store (for logging).
    done_dir: str|Path
        The directory holding the per-region completion markers.
    """
    shutil.rmtree(done_dir, ignore_errors=True)
    logger.info("Zarr store '%s' complete.", zarr_path)


def write_time_region(zarr_path, values, t_offset):
    """
    Write a (time, y, x) block of values into the store's time axis at t_offset.

    Parameters
    ----------
    zarr_path: str|Path
        The path of the initialized zarr store.
    values: np.ndarray
        The (T, Y, X) block covering the full spatial extent of the store.
    t_offset: int
        The index on the store's time axis where the block starts.
    """
    # No coordinate variables: only the 'precip' region is written.
    block = xr.Dataset({'precip': (('time', 'y', 'x'), values)})
    block.to_zarr(str(zarr_path),
                  region={'time': slice(t_offset, t_offset + values.shape[0])},
                  consolidated=False)
