import logging
import multiprocessing
import concurrent.futures
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from swafi.config import Config
from swafi.domain import Domain
from swafi.precip_combiprecip import CombiPrecip
from swafi.precip_combiprecip_5min import CombiPrecip5min
from swafi.utils.precip_reference import load_reference, save_reference

logger = logging.getLogger(__name__)

_n_parts = max(1, int(multiprocessing.cpu_count() * 0.9))

# Tile size [m] for the 5-min extraction. The per-worker memory footprint is
# roughly n_time_steps x (tile_size/1000)^2 x 4 bytes (e.g. 5 years of 5-min
# data on a 32x32 km tile ~ 2.2 GB).
_tile_size_5min = 32000


def detection_tag(detection_window_h=1.0, detection_threshold=None,
                  detection_centered=True, detection_peak_days=True,
                  time_step_h=1.0):
    """Build the file-name tag identifying the event detection settings.

    Caches (part directories, merged parquet, test-event pickles) must be unique
    per detection setting, otherwise events extracted with another definition
    would be silently reused.

    Centring and peak dating only bear on a detection window spanning several
    time steps, so they are left out of the tag when the window is a single
    step: the extractions predating them are then still picked up, instead of
    being re-run to produce the very same events.
    """
    if detection_window_h is None:
        tag = '_detnative'
    elif detection_window_h < 1:
        tag = f"_det{round(detection_window_h * 60)}min"
    else:
        tag = f"_det{detection_window_h:g}h"

    if detection_threshold is not None:
        tag += f"_thr{detection_threshold:g}mm"

    w_det = (1 if detection_window_h is None
             else max(1, int(round(detection_window_h / time_step_h))))
    if w_det > 1:
        if detection_centered:
            tag += "_centered"
        if detection_peak_days:
            tag += "_peakdays"

    return tag


def _get_precipitation(precip_dataset, y_start, y_end, config):
    """Instantiate and open the precipitation source for the given dataset name."""
    if precip_dataset == 'hourly':
        cpc = CombiPrecip(y_start, y_end)
        cpc.open_files(config.get('DIR_PRECIP_HOURLY'))
    elif precip_dataset == '5min':
        # Lazily opened zarr store: only the chunks covering the selected tile
        # are read from disk (the raw zips would require materialising the full
        # grid for every day).
        cpc = CombiPrecip5min(y_start, y_end)
        cpc.open_zarr(config.get('PATH_PRECIP_5MIN_ZARR'))
    else:
        raise ValueError(f"Unknown precipitation dataset: {precip_dataset}")
    return cpc


def process_part(i, part, config, y_start, y_end, method, filter_size, output_dir,
                 precip_dataset='hourly', detection_window_h=1.0, n_parts=None,
                 reference=None, collect_reference=False, detection_threshold=None,
                 detection_centered=True, detection_peak_days=True):
    # The part count is embedded in the file name so that a resume with a
    # different partitioning (e.g. a machine with another CPU count for the
    # hourly split) does not silently reuse parts covering different cells.
    output_file = Path(output_dir) / f"part_{n_parts}_{i}.parquet"
    ref_file = Path(output_dir) / f"part_{n_parts}_{i}_ref.pkl"
    if output_file.exists() and (not collect_reference or ref_file.exists()):
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

    collect = {} if collect_reference else None
    list_of_events = []
    for _, row in part.iterrows():
        events = cpc.extract_events(
            row, method, detection_window_h=detection_window_h,
            detection_threshold=detection_threshold,
            detection_centered=detection_centered,
            detection_peak_days=detection_peak_days,
            reference=reference, collect_reference=collect)
        if events is not None:
            list_of_events.append(events)

    if list_of_events:
        events = pd.concat(list_of_events, axis=0).reset_index(drop=True)
    else:
        events = pd.DataFrame()
    events.to_parquet(output_file)

    if collect_reference:
        with open(ref_file, 'wb') as f:
            pickle.dump(collect, f, protocol=pickle.HIGHEST_PROTOCOL)
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


def _run_workers(parts, config, y_start, y_end, method, filter_size, output_dir,
                 precip_dataset='hourly', detection_window_h=1.0, max_workers=None,
                 reference=None, collect_reference=False, detection_threshold=None,
                 detection_centered=True, detection_peak_days=True):
    n_parts = len(parts)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, part in enumerate(parts):
            # Pass each worker only the references for its own cells, so the
            # full (per-cell) reference is never duplicated across workers.
            part_ref = None
            if reference is not None:
                part_ref = {c: reference[c] for c in part['cid'].to_numpy()
                            if c in reference}
            futures.append(executor.submit(
                process_part, i, part, config, y_start, y_end, method, filter_size,
                output_dir, precip_dataset, detection_window_h, n_parts,
                part_ref, collect_reference, detection_threshold,
                detection_centered, detection_peak_days))
        results = [
            f.result()
            for f in tqdm(concurrent.futures.as_completed(futures),
                          total=len(futures), desc="Parts completed")
        ]
    assert all(results), "Some parts failed to process."


def _merge_parts(output_dir, n_parts):
    return pd.concat(
        [pd.read_parquet(Path(output_dir) / f"part_{n_parts}_{i}.parquet") for i in range(n_parts)],
        ignore_index=True,
    )


def _merge_references(output_dir, n_parts):
    """Merge the per-part reference dicts written by the workers into one
    mapping (cid -> reference dict)."""
    reference = {}
    for i in range(n_parts):
        ref_file = Path(output_dir) / f"part_{n_parts}_{i}_ref.pkl"
        if ref_file.exists():
            with open(ref_file, 'rb') as f:
                reference.update(pickle.load(f))
    return reference


def _stamp_precip_dataset(events, precip_dataset):
    """
    Record the source precipitation dataset as a categorical provenance column
    (survives parquet/pickle/CSV; ~1 byte per row).
    """
    events['precip_dataset'] = pd.Categorical(
        [precip_dataset] * len(events), categories=[precip_dataset])
    return events


def extract_events_parallel(y_start, y_end, method, filter_size=None,
                            precip_dataset='hourly', detection_window_h=1.0, max_workers=None,
                            reference_path=None, save_reference_path=None,
                            detection_threshold=None, detection_centered=True,
                            detection_peak_days=True):
    """Extract events for all domain cells in parallel and return a DataFrame.

    reference_path: str|Path|None
        A per-cell training reference to normalise the events against (see
        Precipitation._extract_events_simple). None re-estimates on this period.
    save_reference_path: str|Path|None
        When given (training extraction), the per-cell reference computed on this
        period is saved to that path for later reuse on the test period.
    detection_threshold: float|None
        An absolute detection threshold [mm] on the detection window (see
        Precipitation._extract_events_simple). None uses the per-cell q98.
    detection_centered: bool
        Whether the detection window is centred on the step it labels.
    detection_peak_days: bool
        Whether to date the events on the intensity peak of each exceeding
        window (see Precipitation._build_peak_event_dates).
    """
    config = Config()
    parts = _split_parts(config, precip_dataset)
    reference = load_reference(reference_path) if reference_path else None
    collect = save_reference_path is not None
    with tempfile.TemporaryDirectory() as tmp_dir:
        _run_workers(parts, config, y_start, y_end, method, filter_size, tmp_dir,
                     precip_dataset, detection_window_h, max_workers, reference,
                     collect, detection_threshold, detection_centered,
                     detection_peak_days)
        events = _merge_parts(tmp_dir, len(parts))
        if collect:
            save_reference(_merge_references(tmp_dir, len(parts)), save_reference_path)
    events = _stamp_precip_dataset(events, precip_dataset)
    logger.info("Extracted %d events for %d-%d.", len(events), y_start, y_end)
    return events


def run_parallel_extraction(y_start, y_end, method, filter_size=None,
                            output_dir="event_parts", output_path='.', precip_dataset='hourly',
                            detection_window_h=1.0, max_workers=None,
                            reference_path=None, save_reference_path=None,
                            detection_threshold=None, detection_centered=True,
                            detection_peak_days=True):
    """Extract events in parallel, saving intermediate parts to output_dir and merging to output_path.

    reference_path / save_reference_path / detection_threshold /
    detection_centered / detection_peak_days: see extract_events_parallel. Training extraction
    passes save_reference_path to persist the per-cell reference; test
    extraction passes reference_path to reuse it.
    """
    config = Config()
    parts = _split_parts(config, precip_dataset)
    os.makedirs(output_dir, exist_ok=True)
    reference = load_reference(reference_path) if reference_path else None
    collect = save_reference_path is not None
    _run_workers(parts, config, y_start, y_end, method, filter_size, output_dir,
                 precip_dataset, detection_window_h, max_workers, reference,
                 collect, detection_threshold, detection_centered,
                 detection_peak_days)
    logger.info("All parts processed. Saved in '%s'.", output_dir)
    events = _merge_parts(output_dir, len(parts))
    events = _stamp_precip_dataset(events, precip_dataset)
    events.to_parquet(output_path)
    logger.info("Merged into '%s'.", output_path)
    if collect:
        reference = _merge_references(output_dir, len(parts))
        save_reference(reference, save_reference_path)
        logger.info("Saved per-cell reference (%d cells) to '%s'.",
                    len(reference), save_reference_path)
