#!/usr/bin/env python
# coding: utf-8

import logging
import os
import multiprocessing
import concurrent.futures
import pandas as pd
import numpy as np
from tqdm import tqdm

from swafi.config import Config
from swafi.domain import Domain
from swafi.precip_combiprecip import CombiPrecip
from swafi.utils.logging_setup import setup_logging

# Configuration for the script
n_cpus = multiprocessing.cpu_count()
n_parts = int(n_cpus * 0.5)  # Number of parts to split the data into for parallel processing

# Definition of events extraction method ('classic' for Bernet et al. 2019 or 'simple'
# for the new simple approach).
METHOD = 'simple'
Y_START = 2005
Y_END = 2024


def process_part(i, part, config):
    # Load precipitation files
    cpc = CombiPrecip(Y_START, Y_END)
    cpc.open_files(config.get('DIR_PRECIP'))

    # Extract coordinates and precipitation data for each part
    cpc.data = cpc.data.sel(
        x=slice(part.x.min() - 5000, part.x.max() + 5000),
        y=slice(part.y.max() + 5000, part.y.min() - 5000)
    )

    # Apply the 3x3km smoothing
    cpc.apply_smoothing(filter_size=3)

    # Apply extract_events() function to all grid cells in part
    list_of_events = []
    for _, row in part.iterrows():
        list_of_events.append(cpc.extract_events(row, METHOD))

    # Store and save data as a .parquet file
    events = pd.concat(list_of_events, axis=0).reset_index(drop=True)
    events.to_parquet(f"event_parts/part_{i}.parquet")

    return True


if __name__ == "__main__":
    setup_logging(script_name='extract_precipitation_events')
    logger = logging.getLogger(__name__)

    config = Config()

    # Get the precipitation data domain
    domain = Domain()
    coords_df = domain.get_coordinates_df()

    # Split the coordinates DataFrame into parts for processing
    indices = np.array_split(np.arange(len(coords_df)), n_parts)
    parts = [coords_df.iloc[idx] for idx in indices]

    # Create a directory to store the event parts
    os.makedirs("event_parts", exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_part, i, part, config) for i, part in enumerate(parts)]
        results = []
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Parts completed"):
            results.append(f.result())
        assert all(results), "Some parts failed to process."

    logger.info("All parts processed successfully. Events saved in 'event_parts/' directory.")

    # Merge all parts into a single DataFrame
    all_events = []
    for i in range(len(parts)):
        part_events = pd.read_parquet(f"event_parts/part_{i}.parquet")
        all_events.append(part_events)
    all_events_df = pd.concat(all_events, ignore_index=True)
    all_events_df.to_parquet(f"events_cpc_model_domain_3x3_{Y_START}_{Y_END}_{METHOD}.parquet")

    logger.info("All parts merged into a single DataFrame.")