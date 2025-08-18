#!/usr/bin/env python
# coding: utf-8

import concurrent.futures
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.ndimage import uniform_filter

from swafi.config import Config
from swafi.domain import Domain
from swafi.precip_combiprecip import CombiPrecip

# Configuration for the script
n_parts = 20  # Number of parts to split the data into for parallel processing
year_start = 2005
year_end = 2023



config = Config()

# Get the precipitation data domain
domain = Domain()
coords_df = domain.get_coordinates_df()

# Load precipitation files
cpc = CombiPrecip(year_start, year_end)
data_full = cpc.open_files(config.get('DIR_PRECIP'))

# Function the calculates the events
def get_events(coords_row, data):
    # Select timeseries and convert it into a DataFrame
    time_series = data.sel(x=coords_row.x, y=coords_row.y).to_dataframe().reset_index()

    # Define parameters for the calculation of the Antecedent Precipitation Index (API)
    n_days = 30
    ts_per_day = 24
    reg = 0.8

    # Calculate the Antecedent Precipitation Index (API) using a convolution
    window = n_days * ts_per_day
    kernel = np.power(reg, np.arange(window) / ts_per_day)
    precip = time_series.precip.values
    api_full = np.convolve(precip, kernel, mode="full")
    time_series["api"] = np.concatenate(([0.0], api_full[:len(precip)-1]))

    # Group events by period of at least 8 hour without precipitation larger than 0.1mm/h and return group IDs
    time_series_th = time_series[time_series.precip >= 0.1]
    group_ids = time_series_th.groupby(
        time_series_th.time.diff().gt("8h").cumsum()).ngroup() + 1

    # Fill gaps between events to correctly calculate all event characteristics and then group again
    time_series["group_ID"] = 0
    time_series.group_ID = group_ids
    ff = time_series.group_ID.ffill()
    bf = time_series.group_ID.bfill()
    time_series.group_ID = ff[ff == bf]
    event_groups = time_series.groupby("group_ID")

    # Get the date and time of the maximum precipitation intensity
    i_max_date = event_groups.apply(lambda g: g.loc[g.precip.idxmax(), 'time'], include_groups=False)

    # Calculate all precipitation characteristics
    events = pd.concat([
        event_groups.time.agg(["first", "last", "size"]),
        event_groups.precip.agg(["sum", "max", "mean", "std"]),
        event_groups.api.first(),
        i_max_date.rename("i_max_date")
    ], axis=1)
    events = events.rename(
        columns={"first": "e_start", "last": "e_end", "size": "duration", "sum": "p_sum",
                 "max": "i_max", "mean": "i_mean", "std": "i_sd", "api": "api", "i_max_date": "i_max_date"})
    events = events.astype({"duration": "int16", "i_sd": "float32", "api": "float32"})

    # Drop events that do not fulfill the condition of minimal precipitation
    events = events[events.p_sum >= 10].reset_index(drop=True)

    # Calculate percentiles of score for each event characteristics
    ranks = events.iloc[:, 2:-1].rank(pct=True).astype("float32")
    ranks.columns = ["duration_q", "p_sum_q", "i_max_q", "i_mean_q", "i_sd_q", "api_q"]
    events = pd.concat([events, ranks], axis=1)

    # Add coordinates to the DataFrame and round all float values
    df_coords = pd.concat([pd.DataFrame(coords_row).T] * len(events), ignore_index=True)
    events = pd.concat([df_coords, events], axis=1).round(5)

    return events


# Split the coordinates DataFrame into parts for processing
parts = np.array_split(coords_df, n_parts)

for i, part in enumerate(parts):

    # Extract coordinates and precipitation data for each part
    data = data_full.sel(x=slice(part.x.min() - 5000, part.x.max() + 5000),
                         y=slice(part.y.max() + 5000, part.y.min() - 5000))
    data.load()

    # Apply the 3x3km smoothing
    data = data.fillna(0)
    data.precip.values = uniform_filter(data.precip, size=(0, 3, 3))

    # Define progress bar
    tq = tqdm(range(0, len(part)), leave=True, position=0)

    # Apply get_events() function to all grid cells in part
    list_of_events = []
    for (index, row), t in zip(part.iterrows(), tq):
        list_of_events.append(get_events(row, data))

    # Store and save data as a .parquet file
    events = pd.concat(list_of_events, axis=0).reset_index(drop=True)
    events.to_parquet(f"event_parts/part_{i}.parquet")
