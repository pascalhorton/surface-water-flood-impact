#!/usr/bin/env python
# coding: utf-8

import os
import multiprocessing
import concurrent.futures
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.ndimage import uniform_filter

from swafi.config import Config
from swafi.domain import Domain
from swafi.precip_combiprecip import CombiPrecip

# Configuration for the script
n_cpus = multiprocessing.cpu_count()
n_parts = int(n_cpus * 0.9)  # Number of parts to split the data into for parallel processing


def process_part(i, part, config):
    # Load precipitation files
    cpc = CombiPrecip()
    cpc.open_files(config.get('DIR_PRECIP'))

    # Extract coordinates and precipitation data for each part
    cpc.data = cpc.data.sel(
        x=slice(part.x.min() - 5000, part.x.max() + 5000),
        y=slice(part.y.max() + 5000, part.y.min() - 5000)
    )

    # Apply the 3x3km smoothing
    cpc.apply_smoothing(filter_size=3)

    # Apply get_events() function to all grid cells in part
    list_of_events = []
    for _, row in part.iterrows():
        list_of_events.append(cpc.extract_events(row))

    # Store and save data as a .parquet file
    events = pd.concat(list_of_events, axis=0).reset_index(drop=True)
    events.to_parquet(f"event_parts/part_{i}.parquet")

    return True


if __name__ == "__main__":
    config = Config()

    # Get the precipitation data domain
    domain = Domain()
    coords_df = domain.get_coordinates_df()

    # Split the coordinates DataFrame into parts for processing
    parts = np.array_split(coords_df, n_parts)

    # Create a directory to store the event parts
    os.makedirs("event_parts", exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_part, i, part, config) for i, part in enumerate(parts)]
        results = []
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Parts completed"):
            results.append(f.result())
        assert all(results), "Some parts failed to process."

    print("All parts processed successfully. Events saved in 'event_parts/' directory.")

    # Merge all parts into a single DataFrame
    all_events = []
    for i in range(len(parts)):
        part_events = pd.read_parquet(f"event_parts/part_{i}.parquet")
        all_events.append(part_events)
    all_events_df = pd.concat(all_events, ignore_index=True)
    all_events_df.to_parquet("events_cpc_model_domain_3x3_2005_2024.parquet")

    print("All parts merged into a single DataFrame.")