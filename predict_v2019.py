from pathlib import Path
from utils.config import Config
import pandas as pd

"""
Compute predictions based on the 2019 version of the data.

The files header are:
- cid: CombiPrecip cell ID
- eid: event ID
- e_start: event start
- e_end: event end
- e_tot: event total duration [h]
- e_tot_q: local quantile of the event duration
- i_max: precipitation maximum intensity [mm/h]
- i_max_q: local quantile of the precipitation maximum intensity
- i_mean: precipitation mean intensity [mm/h]
- i_mean_q: local quantile of the precipitation mean intensity
- i_sd: standard deviation of the precipitation intensity [mm/h]
- i_sd_q: local quantile of the standard deviation of the precipitation intensity
- p_sum: sum of the event precipitation [mm]
- p_sum_q: local quantile of the sum of the event precipitation
- apireg: antecedent precipitation index [mm]
- apireg_q: local quantile of the antecedent precipitation index
- b_pmin: below minimum precipitation threshold of 10 mm [true/false]
- month: month of the year
- seas: season
- wcbeg: weather class at beginning or just before precipitation event
- wcfreq: most frequent weather class during precipitation event (a=ambiguous)
- wc1..i..23: percentage of weather class #i during the precipitation event
- p_evol: temporal evolution of the rainfall event. 4 categories accordingly to the 
  German Association for Water, Wastewater and Waste: continuous (cont; 1), maximum 
  intensity at the beginning (beg; 2), in the middle (mid; 3) or at the end (end; 4) 
  of the rainfall event
[only for event triggered:]
- d_tr_imax: date of the maximum intensity of the triggering event ?
- d_tr_s75: ?
- A: number of damage claims most likely due to SWF
- B: number of damage claims likely due to SWF
- C: number of damage claims due to fluvial flood or SWF
- D: number of damage claims likely due to fluvial flood
- E: number of damage claims most likely due to fluvial flood
- AB: number of claims of A+B together -> SWF
- DE: number of claims of D+E together -> FF
- ABCDE: number of claims per event in grid cell

"""

config = Config(output_dir='predict_v2019')
output_dir = config.output_dir

df_trigg = pd.read_csv(config.get('CSV_2019_TRIGGER'))

