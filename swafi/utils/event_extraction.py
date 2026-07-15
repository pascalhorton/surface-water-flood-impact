import logging
import multiprocessing
import concurrent.futures
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from swafi.config import Config
from swafi.domain import Domain
from swafi.precip_combiprecip import CombiPrecip
from swafi.precip_combiprecip_5min import CombiPrecip5min

logger = logging.getLogger(__name__)

_n_parts = max(1, int(multiprocessing.cpu_count() * 0.9))

# Tile size [m] for the 5-min extraction. The per-worker memory footprint is
# roughly n_time_steps x (tile_size/1000)^2 x 4 bytes (e.g. 5 years of 5-min
# data on a 32x32 km tile ~ 2.2 GB).
_tile_size_5min = 32000


def _get_precipitation(precip_dataset, y_start, y_end, config):
    """Instantiate and open the precipitation source for the given dataset name."""
    if precip_dataset == 'hourly':
        cpc = CombiPrecip(y_start, y_end)
        cpc.open_files(config.get('DIR_PRECIP'))
    elif precip_dataset == '5min':
        # Lazily opened zarr store: only the chunks covering the selected tile
        # are read from disk (the raw zips would require materialising the full
        # grid for every day).
        cpc = CombiPrecip5min(y_start, y_end)
        cpc.open_zarr(config.get('PATH_PRECIP_5MIN_ZARR'))
    else:
        raise ValueError(f"Unknown precipitation dataset: {precip_dataset}")
    return cpc


def process_part(i, part, config, y_start, y_end, method, simple_strict_mode, filter_size, output_dir, precip_dataset='hourly'):
    output_file = Path(output_dir) / f"part_{i}.parquet"
    if output_file.exists():
        logger.info(f"Output file '{output_file}' already exists.")
        return True

    cpc = _get_precipitation(precip_dataset, y_start, y_end, config)
    # Pad the slice only when smoothing needs the neighbouring cells.
    pad = 5000 if filter_size is not None else 0
    cpc.data = cpc.data.sel(
        x=slice(part.x.min() - pad, part.x.max() + pad),
        y=slice(part.y.max() + pad, part.y.min() - pad),
    )
    if filter_size is not None:
        cpc.apply_smoothing(filter_size=filter_size)
    cpc.data = cpc.data.compute()
    list_of_events = [cpc.extract_events(row, method, simple_strict_mode) for _, row in part.iterrows()]
    events = pd.concat(list_of_events, axis=0).reset_index(drop=True)
    events.to_parquet(output_file)
    return True


def _split_coords(config):
    domain = Domain()
    coords_df = domain.get_coordinates_df()
    indices = np.array_split(np.arange(len(coords_df)), _n_parts)
    return [coords_df.iloc[idx] for idx in indices]


def _split_coords_tiles(tile_size=_tile_size_5min):
    """Split the domain cells into square spatial tiles (deterministic order)."""
    domain = Domain()
    coords_df = domain.get_coordinates_df()
    tiles = coords_df.groupby([coords_df.x // tile_size, coords_df.y // tile_size])
    return [part for _, part in tiles]


def _split_parts(config, precip_dataset):
    """Split the domain cells into parts suited to the precipitation dataset."""
    if precip_dataset == '5min':
        # Compact tiles keep the per-worker time series small: a 1-D index
        # split would produce bands spanning the full domain width, which do
        # not fit in memory at the 5-min resolution.
        return _split_coords_tiles()
    return _split_coords(config)


def _run_workers(parts, config, y_start, y_end, method, simple_strict_mode, filter_size, output_dir, precip_dataset='hourly', max_workers=None):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_part, i, part, config, y_start, y_end, method, simple_strict_mode, filter_size, output_dir, precip_dataset)
            for i, part in enumerate(parts)
        ]
        results = [
            f.result()
            for f in tqdm(concurrent.futures.as_completed(futures),
                          total=len(futures), desc="Parts completed")
        ]
    assert all(results), "Some parts failed to process."


def _merge_parts(output_dir, n_parts):
    return pd.concat(
        [pd.read_parquet(Path(output_dir) / f"part_{i}.parquet") for i in range(n_parts)],
        ignore_index=True,
    )


def extract_events_parallel(y_start, y_end, method, simple_strict_mode=False, filter_size=None,
                            precip_dataset='hourly', max_workers=None):
    """Extract events for all domain cells in parallel and return a DataFrame."""
    config = Config()
    parts = _split_parts(config, precip_dataset)
    with tempfile.TemporaryDirectory() as tmp_dir:
        _run_workers(parts, config, y_start, y_end, method, simple_strict_mode, filter_size, tmp_dir, precip_dataset, max_workers)
        events = _merge_parts(tmp_dir, len(parts))
    logger.info("Extracted %d events for %d-%d.", len(events), y_start, y_end)
    return events


def run_parallel_extraction(y_start, y_end, method, simple_strict_mode=False, filter_size=None,
                            output_dir="event_parts", output_path='.', precip_dataset='hourly',
                            max_workers=None):
    """Extract events in parallel, saving intermediate parts to output_dir and merging to output_path."""
    config = Config()
    parts = _split_parts(config, precip_dataset)
    os.makedirs(output_dir, exist_ok=True)
    _run_workers(parts, config, y_start, y_end, method, simple_strict_mode, filter_size, output_dir, precip_dataset, max_workers)
    logger.info("All parts processed. Saved in '%s'.", output_dir)
    events = _merge_parts(output_dir, len(parts))
    events.to_parquet(output_path)
    logger.info("Merged into '%s'.", output_path)
