#!/usr/bin/env python
# coding: utf-8

import logging

from swafi.utils.logging_setup import setup_logging
from swafi.utils.event_extraction import run_parallel_extraction

METHOD = 'simple'
STRICT = True
Y_START = 2005
Y_END = 2024

if __name__ == "__main__":
    setup_logging(script_name='extract_precipitation_events')
    logger = logging.getLogger(__name__)

    output_path = f"events_cpc_model_domain_{Y_START}_{Y_END}_{METHOD}"
    output_path += ".parquet"
    run_parallel_extraction(
        Y_START,
        Y_END,
        METHOD,
        simple_strict_mode=STRICT,
        filter_size=None,
        output_path=output_path
    )
