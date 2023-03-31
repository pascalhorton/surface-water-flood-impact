#!/usr/bin/env python
# coding: utf-8

import xarray as xr
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.ndimage import uniform_filter

# Get quantile data to select the coordinates for the event calculation
ds_q = xr.open_dataset("quantiles_markus/quantiles_0513_markus.nc")

# Get mask
msk = np.isfinite(ds_q.psum_q98.values)

# Get coordinates of the grid cells in Switzerland
coords = ds_q.cid.where(msk).to_dataframe().reset_index()
coords = coords[coords.cid.notnull()].reset_index(drop=True).astype("int32")
coords = pd.concat([coords.iloc[:, -1], coords.iloc[:, :2]], axis=1)

# Load CombiPrecip files
files = sorted(glob("/scratch3/severin/data/meteoswiss/cpch/*"))
datasets = xr.open_mfdataset(files, parallel=True)

# Function the calculates the events
def get_events(coords_row):
    # Select timeseries and convert it into a DataFrame
    time_series = data.sel(x=coords_row.x, y=coords_row.y).to_dataframe().reset_index()

    # Define parameters for the calculation of the Antecedent Precipitation Index (API)
    n_days = 30
    tstepsperday = 24
    reg = 0.8

    # Calculate windows and coefficients for API
    rolled_windows = np.lib.stride_tricks.sliding_window_view(
        np.r_[np.zeros(n_days * tstepsperday), time_series.CPC.values],
        n_days * tstepsperday)
    reg_coef = np.power(np.repeat(reg, n_days * tstepsperday),
                        (np.arange(n_days * tstepsperday) / tstepsperday)[::-1])

    # The API for each step is given by the product of the coefficients and the windows
    time_series["api"] = np.matmul(rolled_windows, reg_coef)[:-1]

    # Group events by period of at least 8 hour without precipitation larger than 0.1mm/h and return group IDs
    time_series_th = time_series[time_series.CPC >= 0.1]
    group_ids = time_series_th.groupby(
        time_series_th.time.diff().gt("8h").cumsum()).ngroup() + 1

    # Fill gaps between events to correctly calculate all event characteristics and then group again
    time_series["group_ID"] = 0
    time_series.group_ID = group_ids
    ff = time_series.group_ID.ffill()
    bf = time_series.group_ID.bfill()
    time_series.group_ID = ff[ff == bf]
    event_groups = time_series.groupby("group_ID")

    # Calculate all precipitation characteristics
    events = pd.concat([event_groups.time.agg(["first", "last", "size"]),
                        event_groups.CPC.agg(["sum", "max", "mean", "std"]),
                        event_groups.api.first()], axis=1)
    events = events.rename(
        columns={"first": "e_start", "last": "e_end", "size": "e_tot", "sum": "p_sum",
                 "max": "i_max", "mean": "i_mean", "std": "i_sd", "api": "apireg"})
    events = events.astype({"e_tot": "int16", "i_sd": "float32", "apireg": "float32"})

    # Drop events that do not fulfill the condition of minimal precipitation
    events = events[events.p_sum >= 10].reset_index(drop=True)

    # Calculate percentiles of score for each event characteristics
    ranks = events.iloc[:, 2:].rank(pct=True).astype("float32")
    ranks.columns = ["e_tot_q", "p_sum_q", "i_max_q", "i_mean_q", "i_sd_q", "apireg_q"]
    events = pd.concat([events, ranks], axis=1)

    # Add coordinates to the DataFrame and round all float values
    df_coords = pd.concat([pd.DataFrame(coords_row).T] * len(events), ignore_index=True)
    events = pd.concat([df_coords, events], axis=1).round(5)

    return events


# Define steps for 10 parts, on which the event calculation is applied
steps = np.arange(0, len(coords), 4600)
steps[-1] = len(coords)

# Define file names
parquet_files = ["part" + str(i) for i in range(1, len(steps))]

for i in range(1, len(steps) - 1):

    # Extract coordinates and CombiPrecip data for each part
    part = coords[steps[i]:steps[i + 1]]
    data = datasets.sel(x=slice(part.x.min() - 5000, part.x.max() + 5000),
                        y=slice(part.y.max() + 5000, part.y.min() - 5000))
    data.load()

    # Apply the 3x3km smoothing
    data = data.fillna(0)
    data.CPC.values = uniform_filter(data.CPC, size=(0, 3, 3))

    # Define progress bar
    tq = tqdm(range(0, len(part)), leave=True, position=0)

    # Apply get_events() function to all grid cells in part
    list_of_events = []
    for (index, row), t in zip(part.iterrows(), tq):
        list_of_events.append(get_events(row))

    # Store and save data as a .parquet file
    events = pd.concat(list_of_events, axis=0).reset_index(drop=True)
    events.to_parquet("event_parts/" + parquet_files[i] + ".parquet")
