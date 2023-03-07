from pathlib import Path
from utils.config import Config
from utils.spatial_operations import rasterize
import numpy as np
import richdem as rd

"""
Computes the flow accumulation (in m2) for all files listed under "DEM_PATHS" in the
config file. Save the results as tiff file.
"""

config = Config(output_dir='prepare_flowacc')
output_dir = config.output_dir

# Get file paths
DEM_PATHS = config.get('DEM_PATHS')

# Compute the flow accumulation for each file
for dem_path in DEM_PATHS:

    filepath_flowacc = f'{output_dir}/{Path(dem_path).stem}_flowacc.tif'
    filepath_nolakes = f'{output_dir}/{Path(dem_path).stem}_flowacc_nolakes.tif'
    filepath_norivers = f'{output_dir}/{Path(dem_path).stem}_flowacc_norivers.tif'

    flowacc = None

    # Compute flow accumulation
    if not Path(filepath_flowacc).exists():
        dem = rd.LoadGDAL(dem_path, no_data=-9999.0)
        cell_area = abs(dem.geotransform[1] * dem.geotransform[5])

        # Fill depressions
        rd.FillDepressions(dem, epsilon=True, in_place=True)

        # Compute flow accumulation
        flowacc = rd.FlowAccumulation(dem, method='Dinf')
        flowacc[flowacc == -1] = np.nan
        flowacc *= cell_area

        # Save to file
        rd.SaveGDAL(filepath_flowacc, flowacc)
    else:
        flowacc = rd.LoadGDAL(filepath_flowacc)

    # Mask out lakes
    if not Path(filepath_nolakes).exists():
        # Remove lakes
        mask_lakes = rasterize(config.get('LAKES_PATH'), dem_path)
        flowacc[mask_lakes == 1] = np.nan

        # Save to file
        rd.SaveGDAL(filepath_nolakes, flowacc)
    else:
        flowacc = rd.LoadGDAL(filepath_nolakes)

    # Mask out rivers
    if not Path(filepath_norivers).exists():
        # Remove flow accumulation superior to the given threshold
        flowacc[flowacc >= config.get('FLOWACC_RIVER_THRESHOLD', 50000)] = np.nan

        # Save to file
        rd.SaveGDAL(filepath_norivers, flowacc)
    else:
        flowacc = rd.LoadGDAL(filepath_norivers)

