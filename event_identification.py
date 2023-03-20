"""
This script extracts precipitation events following the definition given in Bernet et al. 2019
"""


#import packages
import xarray as xr
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
from scipy.ndimage import uniform_filter

#get quantile data to select the coordinates for the event calculation
ds_q = xr.open_dataset("quantiles_markus/quantiles_0513_markus.nc")

#get mask
msk = np.isfinite(ds_q.psum_q98.values)

#get coordinates of the grid cells in Switzerland
coords = ds_q.cid.where(msk).to_dataframe().reset_index()
coords = coords[coords.cid.notnull()].reset_index(drop=True).astype("int32")
coords = pd.concat([coords.iloc[:,-1], coords.iloc[:,:2]], axis =1)

#load CombiPrecip files 
files = sorted(glob("/scratch3/severin/data/meteoswiss/cpch/*"))
datasets = xr.open_mfdataset(files, parallel = True)


#define function the calculates the events
def get_events(coords_row):
    
    #select timeseries and convert it into a DataFrame
    time_series = data.sel(x=coords_row.x, y=coords_row.y).to_dataframe().reset_index()
    
    #define parameters for the calculation of the Antecedent Precipitation Index (API) following the definition in Bernet et al. 2019
    n_days = 30
    tstepsperday = 24
    reg = 0.8

    #calculate windows and coefficients for the API
    rolled_windows = np.lib.stride_tricks.sliding_window_view(np.r_[np.zeros(n_days*tstepsperday),time_series.CPC.values], n_days*tstepsperday)
    reg_coef = np.power(np.repeat(reg, n_days*tstepsperday), (np.arange(n_days*tstepsperday)/tstepsperday)[::-1])

    #the API for each step is given by the product of the coefficients and the windows
    time_series["api"] = np.matmul(rolled_windows,reg_coef)[:-1]
    
    #group events by period of at least 8 time steps without precipitation larger than 0.1mm/h
    time_series_th = time_series[time_series.CPC >= 0.1]
    groups = time_series_th.groupby(time_series_th.time.diff().ge("8h").cumsum())
    
    #calculate several characteristics for each event and store data in an DataFrame
    events = pd.concat([groups.time.agg(["first", "last"]), groups.CPC.agg(["max", "sum","mean", "std"]), groups.api.first()], axis = 1)
    events = events.rename(columns = {"first":"start", "last":"end", "max":"max_int", "sum":"agg_sum", "mean":"mean_int", "std":"std_int"})
    events = events.astype({"max_int":"float32", "agg_sum":"float32", "std_int":"float32", "api":"float32"})
    
    #drop events that do not fulfill the condition of minimal precipitation
    events = events[events.agg_sum>=10].reset_index(drop=True)
    
    #calculate duration
    events["duration"] = events.end - events.start
    
    #add coordinates to the DataFrame
    df_coords = pd.concat([pd.DataFrame(coords_row).T]*len(events), ignore_index = True)
    events = pd.concat([df_coords, events], axis = 1)
    
    return events


#define steps for 10 parts, on which the event calculation is applied
steps = np.arange(0,len(coords), 4600)
steps[-1] = len(coords)

#define file names
parquet_files = ["part"+str(i) for i in range(1,len(steps))]

for i in range(1,len(steps)-1): 
    
    #extract coordinates and CombiPrecip data for each part
    part = coords[steps[i]:steps[i+1]]
    data = datasets.sel(x = slice(part.x.min()-5000, part.x.max()+5000), y = slice(part.y.max()+5000, part.y.min()-5000))
    data.load()
    
    #apply the 3x3km smoothing
    data = data.fillna(0)
    data.CPC.values = uniform_filter(data.CPC,size=(0,3,3))
    
    #define progress bar
    tq = tqdm(range(0, len(part)), leave = True, position = 0)

    #apply get_events() function to all grid cells in part
    list_of_events = []
    for (index,row),t in zip(part.iterrows(), tq):
        list_of_events.append(get_events(row))

    #store and save data as a .parquet file
    events = pd.concat(list_of_events, axis = 0).reset_index(drop=True)
    events.to_parquet("event_parts/"+parquet_files[i]+".parquet")

