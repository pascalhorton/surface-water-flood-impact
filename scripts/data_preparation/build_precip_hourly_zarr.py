#!/usr/bin/env python
# coding: utf-8

"""
Build the hourly CombiPrecip zarr store (one-time conversion of the netCDF
files from DIR_PRECIP). The store replaces the former monthly pickle files:
extraction, statistics and training read small chunks from it lazily.

The build is resumable: rerun the script to continue after an interruption.
"""

import logging

from swafi.config import Config
from swafi.precip_combiprecip import CombiPrecip
from swafi.utils.logging_setup import setup_logging

Y_START = 2005
Y_END = 2024

if __name__ == "__main__":
    setup_logging(script_name='build_precip_hourly_zarr')
    logger = logging.getLogger(__name__)

    config = Config()
    zarr_path = config.get('PATH_PRECIP_HOURLY_ZARR')

    cpc = CombiPrecip(Y_START, Y_END)
    cpc.build_zarr_store(zarr_path)
