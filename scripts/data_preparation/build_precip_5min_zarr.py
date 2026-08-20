#!/usr/bin/env python
# coding: utf-8

import logging

from swafi.config import Config
from swafi.precip_combiprecip_5min import CombiPrecip5min
from swafi.utils.logging_setup import setup_logging

Y_START = 2005
Y_END = 2024
N_WORKERS = 24  # Memory use stays low (~1 day of full-grid data per worker)

if __name__ == "__main__":
    setup_logging(script_name='build_precip_5min_zarr')
    logger = logging.getLogger(__name__)

    config = Config()
    zarr_path = config.get('PATH_PRECIP_5MIN_ZARR')

    cpc = CombiPrecip5min(Y_START, Y_END)
    cpc.build_zarr_store(zarr_path, n_workers=N_WORKERS)
