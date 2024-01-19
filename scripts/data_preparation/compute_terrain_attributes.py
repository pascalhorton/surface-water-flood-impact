"""
Computes terrain attributes for all files listed under "DEM_PATHS" in the config file.
Save the results as tiff file.
"""

from pathlib import Path
import richdem as rd

from swafi.config import Config


config = Config(output_dir='static_terrain')
output_dir = config.output_dir

# Get file paths
DEM_PATHS = config.get('DEM_PATHS')

# Compute the flow accumulation for each file
for dem_path in DEM_PATHS:

    dem = rd.LoadGDAL(dem_path, no_data=-9999.0)

    filepath_slope = f'{output_dir}/{Path(dem_path).stem}_slope.tif'
    filepath_aspect = f'{output_dir}/{Path(dem_path).stem}_aspect.tif'
    filepath_curv_prof = f'{output_dir}/{Path(dem_path).stem}_curv_prof.tif'
    filepath_curv_plan = f'{output_dir}/{Path(dem_path).stem}_curv_plan.tif'
    filepath_curv_tot = f'{output_dir}/{Path(dem_path).stem}_curv_tot.tif'

    # Compute slope
    if not Path(filepath_slope).exists():
        slope = rd.TerrainAttribute(dem, attrib='slope_riserun')
        rd.SaveGDAL(filepath_slope, slope)

    # Compute aspect
    if not Path(filepath_aspect).exists():
        aspect = rd.TerrainAttribute(dem, attrib='aspect')
        rd.SaveGDAL(filepath_aspect, aspect)

    # Compute profile curvature
    if not Path(filepath_curv_prof).exists():
        curv_prof = rd.TerrainAttribute(dem, attrib='profile_curvature')
        rd.SaveGDAL(filepath_curv_prof, curv_prof)

    # Compute planform curvature
    if not Path(filepath_curv_plan).exists():
        curv_plan = rd.TerrainAttribute(dem, attrib='planform_curvature')
        rd.SaveGDAL(filepath_curv_plan, curv_plan)

    # Compute total curvature
    if not Path(filepath_curv_tot).exists():
        curv_tot = rd.TerrainAttribute(dem, attrib='curvature')
        rd.SaveGDAL(filepath_curv_tot, curv_tot)
