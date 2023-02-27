from pathlib import Path
from utils.config import Config
import numpy as np
import richdem as rd

"""
Computes the flow accumulation (in m2) for all files listed under "DEM_PATHS" in the
config file. Save the results as tiff file.
"""

config = Config(output_dir='prepare_flowacc')

# Get file paths
DEM_PATHS = config.get('DEM_PATHS')

# Compute the flow accumulation for each file
for dem_path in DEM_PATHS:

    # Compute flow accumulation
    filename_flowacc = Path(dem_path).stem + '_flowacc.tif'
    filepath_flowacc = f'{config.output_dir}/{filename_flowacc}'
    if not Path.exists(filepath_flowacc):
        dem = rd.LoadGDAL(dem_path, no_data=-9999.0)
        cell_area = abs(dem.geotransform[1] * dem.geotransform[5])

        # Fill depressions
        rd.FillDepressions(dem, epsilon=True, in_place=True)

        # Compute flow accumulation
        accum_dinf = rd.FlowAccumulation(dem, method='Dinf')
        accum_dinf[accum_dinf == -1] = np.nan
        accum_dinf *= cell_area

        # Save to file
        rd.SaveGDAL(filepath_flowacc, accum_dinf)
