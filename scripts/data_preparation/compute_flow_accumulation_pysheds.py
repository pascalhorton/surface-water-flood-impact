"""
Computes the flow accumulation (in m2) for all files listed under "DEM_PATHS" in the
config file using PySheds. Save the results as tiff file.
"""

from pathlib import Path
import numpy as np
from pysheds.grid import Grid

from swafi.config import Config
from swafi.utils.spatial import rasterize


config = Config(output_dir='static_flowacc')
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
        grid = Grid.from_raster(dem_path)
        dem = grid.read_raster(dem_path)
        dem[dem == -9999] = np.nan

        # Cell area
        cell_area = abs(grid.affine.a * grid.affine.e)

        # Fill pits in DEM
        pit_filled_dem = grid.fill_pits(dem)

        # Fill depressions in DEM
        flooded_dem = grid.fill_depressions(pit_filled_dem)

        # Resolve flats in DEM
        inflated_dem = grid.resolve_flats(flooded_dem)

        # Compute flow direction
        fdir = grid.flowdir(inflated_dem, routing='dinf')

        # Calculate flow accumulation
        flowacc = grid.accumulation(fdir, routing='dinf')
        flowacc *= cell_area
        flowacc[np.isnan(dem)] = np.nan

        # Save to file
        grid.to_raster(flowacc, filepath_flowacc)
    else:
        grid = Grid.from_raster(filepath_flowacc)
        flowacc = grid.read_raster(filepath_flowacc)

    # Mask out lakes
    if not Path(filepath_nolakes).exists():
        # Remove lakes
        mask_lakes = rasterize(config.get('LAKES_PATH'), dem_path)
        flowacc[mask_lakes == 1] = np.nan

        # Save to file
        grid.to_raster(flowacc, filepath_nolakes)
    else:
        grid = Grid.from_raster(filepath_nolakes)
        flowacc = grid.read_raster(filepath_nolakes)

    # Mask out rivers
    if not Path(filepath_norivers).exists():
        # Remove flow accumulation superior to the given threshold
        flowacc[flowacc >= config.get('FLOWACC_RIVER_THRESHOLD', 50000)] = np.nan

        # Save to file
        grid.to_raster(flowacc, filepath_norivers)
    else:
        grid = Grid.from_raster(filepath_norivers)
        flowacc = grid.read_raster(filepath_norivers)
