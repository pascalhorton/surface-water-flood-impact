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

logger = logging.getLogger(__name__)

_n_parts = max(1, int(multiprocessing.cpu_count() * 0.5))


def process_part(i, part, config, y_start, y_end, method, output_dir):
    cpc = CombiPrecip(y_start, y_end)
    cpc.open_files(config.get('DIR_PRECIP'))
    cpc.data = cpc.data.sel(
        x=slice(part.x.min() - 5000, part.x.max() + 5000),
        y=slice(part.y.max() + 5000, part.y.min() - 5000),
    )
    cpc.apply_smoothing(filter_size=3)
    list_of_events = [cpc.extract_events(row, method) for _, row in part.iterrows()]
    events = pd.concat(list_of_events, axis=0).reset_index(drop=True)
    events.to_parquet(Path(output_dir) / f"part_{i}.parquet")
    return True


def _split_coords(config):
    domain = Domain()
    coords_df = domain.get_coordinates_df()
    indices = np.array_split(np.arange(len(coords_df)), _n_parts)
    return [coords_df.iloc[idx] for idx in indices]


def _run_workers(parts, config, y_start, y_end, method, output_dir):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_part, i, part, config, y_start, y_end, method, output_dir)
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


def extract_events_parallel(y_start, y_end, method):
    """Extract events for all domain cells in parallel and return a DataFrame."""
    config = Config()
    parts = _split_coords(config)
    with tempfile.TemporaryDirectory() as tmp_dir:
        _run_workers(parts, config, y_start, y_end, method, tmp_dir)
        events = _merge_parts(tmp_dir, len(parts))
    logger.info("Extracted %d events for %d-%d.", len(events), y_start, y_end)
    return events


def run_parallel_extraction(y_start, y_end, method, output_dir, output_path):
    """Extract events in parallel, saving intermediate parts to output_dir and merging to output_path."""
    config = Config()
    parts = _split_coords(config)
    os.makedirs(output_dir, exist_ok=True)
    _run_workers(parts, config, y_start, y_end, method, output_dir)
    logger.info("All parts processed. Saved in '%s'.", output_dir)
    events = _merge_parts(output_dir, len(parts))
    events.to_parquet(output_path)
    logger.info("Merged into '%s'.", output_path)
