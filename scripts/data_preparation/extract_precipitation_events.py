#!/usr/bin/env python
# coding: utf-8

import logging
from pathlib import Path

from swafi.config import Config
from swafi.utils.logging_setup import setup_logging
from swafi.utils.event_extraction import run_parallel_extraction

PRECIP_DATASET = 'hourly'  # 'hourly' (CombiPrecip netCDF) or '5min' (zarr store)
METHOD = 'simple'
STRICT = True  # Extract only days exceeding the threshold. Recommended.
Y_START = 2005
Y_END = 2024
MAX_WORKERS = 10  # Memory ~ MAX_WORKERS x part footprint (~2.2 GB/tile for 5 years of 5-min data)

if __name__ == "__main__":
    setup_logging(script_name='extract_precipitation_events')
    logger = logging.getLogger(__name__)

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
        output_dir = 'event_parts_5min'
    else:
        dataset_tag = 'cpc'
        output_dir = 'event_parts'

    output_path = f"events_{dataset_tag}_model_domain_{Y_START}_{Y_END}_{METHOD}.parquet"
    run_parallel_extraction(
        Y_START,
        Y_END,
        METHOD,
        simple_strict_mode=STRICT,
        filter_size=None,
        output_dir=output_dir,
        output_path=output_path,
        precip_dataset=PRECIP_DATASET,
        max_workers=MAX_WORKERS
    )
