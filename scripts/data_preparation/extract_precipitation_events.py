#!/usr/bin/env python
# coding: utf-8

import logging
from pathlib import Path

from swafi.config import Config
from swafi.utils.logging_setup import setup_logging
from swafi.utils.event_extraction import detection_tag, run_parallel_extraction

PRECIP_DATASET = 'hourly'  # 'hourly' (CombiPrecip netCDF) or '5min' (zarr store)
METHOD = 'simple'
DETECTION_WINDOW_H = 1  # Accumulation window [h] for the detection threshold (None = native time step)
# Absolute detection threshold [mm] on that accumulation, e.g. DETECTION_WINDOW_H = 12
# with DETECTION_THRESHOLD = 10 selects the days reaching p_12h >= 10mm. None uses
# the per-cell q98 of the accumulation window (relative, period-dependent) instead.
DETECTION_THRESHOLD = None
# Centre the detection window on the step it labels, and date the events on the
# intensity peak of each exceeding window rather than on the exceedances
# themselves (which counts a storm once per day its window slides over).
# Both are no-ops when the window is a single time step.
DETECTION_CENTERED = True
DETECTION_PEAK_DAYS = True
Y_START = 2005
Y_END = 2024
MAX_WORKERS = 10  # Memory ~ MAX_WORKERS x part footprint (~2.2 GB/tile for 5 years of 5-min data)

if __name__ == "__main__":
    setup_logging(script_name='extract_precipitation_events')
    logger = logging.getLogger(__name__)

    if METHOD == 'classic' and PRECIP_DATASET != 'hourly':
        raise ValueError("The classic method relies on hourly data.")

    if PRECIP_DATASET == '5min':
        config = Config()
        zarr_path = config.get('PATH_PRECIP_5MIN_ZARR', do_raise=False)
        if not zarr_path or not Path(zarr_path).exists():
            where = f"'{zarr_path}'" if zarr_path else "(PATH_PRECIP_5MIN_ZARR not set)"
            raise FileNotFoundError(
                f"The 5-min zarr store {where} does not exist. Build it first "
                f"with scripts/data_preparation/build_precip_5min_zarr.py (config key "
                f"PATH_PRECIP_5MIN_ZARR).")
        dataset_tag = 'cpc_5min'
    else:
        # Explicit dataset tag for the simple method; the classic method is
        # hourly by definition and stays untagged.
        dataset_tag = 'cpc_hourly' if METHOD == 'simple' else 'cpc'

    # Detection tag (simple method only: the classic method does not use the
    # detection threshold window)
    if METHOD != 'simple':
        det_tag = ''
    else:
        det_tag = detection_tag(DETECTION_WINDOW_H, DETECTION_THRESHOLD,
                                DETECTION_CENTERED, DETECTION_PEAK_DAYS,
                                5 / 60 if PRECIP_DATASET == '5min' else 1.0)

    # The parts directory is a resumable cache: it must be unique per
    # configuration, otherwise parts from another run would be reused.
    output_dir = f"event_parts_{dataset_tag}_{METHOD}{det_tag}"
    output_path = f"events_{dataset_tag}_model_domain_{Y_START}_{Y_END}_{METHOD}{det_tag}.parquet"

    # For the simple method, persist the per-cell normalisation reference (q98
    # threshold + CDFs) so that events extracted over other (test) periods can be
    # ranked against this training distribution instead of their own. The classic
    # method defines events by absolute thresholds and needs no reference.
    save_reference_path = None
    if METHOD == 'simple':
        save_reference_path = output_path.replace('.parquet', '_ref.pkl')

    run_parallel_extraction(
        Y_START,
        Y_END,
        METHOD,
        filter_size=None,
        output_dir=output_dir,
        output_path=output_path,
        precip_dataset=PRECIP_DATASET,
        detection_window_h=DETECTION_WINDOW_H,
        detection_threshold=DETECTION_THRESHOLD,
        detection_centered=DETECTION_CENTERED,
        detection_peak_days=DETECTION_PEAK_DAYS,
        max_workers=MAX_WORKERS,
        save_reference_path=save_reference_path,
    )
