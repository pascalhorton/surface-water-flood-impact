"""
Computes the topographic wetness index (TWI) for the provided files.
"""

from pathlib import Path
import numpy as np
import rasterio

from swafi.config import Config


config = Config(output_dir='static_twi')
base_dir = config.get('OUTPUT_DIR')
output_dir = config.output_dir

# Get file paths
FLOWACC_FILES = [
    base_dir + '/static_flowacc_pysheds/dem_010m_flowacc_nolakes.tif',
    base_dir + '/static_flowacc_pysheds/dem_025m_flowacc_nolakes.tif',
    base_dir + '/static_flowacc_pysheds/dem_050m_flowacc_nolakes.tif'
]
SLOPE_FILES = [
    base_dir + '/static_terrain/dem_010m_slope.tif',
    base_dir + '/static_terrain/dem_025m_slope.tif',
    base_dir + '/static_terrain/dem_050m_slope.tif'
]
RES_FILES = [
    output_dir / 'dem_010m_twi.tif',
    output_dir / 'dem_025m_twi.tif',
    output_dir / 'dem_050m_twi.tif'
]

# Compute the topographic wetness index for each pair of file
for files in zip(FLOWACC_FILES, SLOPE_FILES, RES_FILES):
    flowacc_file, slope_file, res_file = files

    # Compute flow accumulation
    if not Path(res_file).exists():
        with rasterio.open(slope_file) as slope_ds:
            slope_data = slope_ds.read(1)

        with rasterio.open(flowacc_file) as flow_acc_ds:
            flow_acc_data = flow_acc_ds.read(1)

        # Convert slope to radians
        slope_data_radians = np.arctan(slope_data)

        # Compute TWI
        twi_data = np.log(flow_acc_data / np.tan(slope_data_radians))

        # Limit TWI to 0-30
        twi_data[twi_data < 0] = 0
        twi_data[twi_data > 30] = 30

        # Create and save the TWI raster
        with rasterio.open(res_file, 'w', driver='GTiff', width=slope_ds.width,
                           height=slope_ds.height, count=1, dtype=twi_data.dtype,
                           crs=slope_ds.crs, transform=slope_ds.transform) as twi_ds:
            twi_ds.write(twi_data, 1)
