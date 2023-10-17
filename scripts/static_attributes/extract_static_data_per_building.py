"""
Extract static data from geotiff files for each building from the TLM geopackage.
"""

import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import numpy as np
from swafi.config import Config

config = Config(output_dir='static_attributes_buildings')
base_dir = config.get('OUTPUT_DIR')

# Select the attributes of interest.
attribute_files = [
    base_dir + '/static_terrain/dem_010m_curv_plan.tif',
    base_dir + '/static_terrain/dem_010m_slope.tif',
    base_dir + '/static_terrain/dem_025m_curv_plan.tif',
    base_dir + '/static_terrain/dem_025m_slope.tif',
    base_dir + '/static_terrain/dem_050m_slope.tif',
    base_dir + '/static_flowacc_pysheds/dem_010m_flowacc_nolakes.tif',
    base_dir + '/static_flowacc_pysheds/dem_025m_flowacc_nolakes.tif',
    base_dir + '/static_flowacc_pysheds/dem_050m_flowacc_nolakes.tif',
    base_dir + '/static_twi/dem_010m_twi.tif',
    base_dir + '/static_twi/dem_025m_twi.tif',
    base_dir + '/static_twi/dem_050m_twi.tif',
]

attribute_names = [
    '010m_curv_plan',
    '010m_slope',
    '025m_curv_plan',
    '025m_slope',
    '050m_slope',
    '010m_flowacc',
    '025m_flowacc',
    '050m_flowacc',
    '010m_twi',
    '025m_twi',
    '050m_twi',
]


def main():
    # Read the GeoPackage and raster data
    gdf = gpd.read_file(config.get('BUILDINGS_GPKG'),
                        layer=config.get('BUILDINGS_LAYER'))

    # Add the attribute columns to the GeoDataFrame
    for attribute_name in attribute_names:
        gdf[attribute_name] = np.nan

    # Loop through each attribute file
    for attribute_file, attribute_name in zip(attribute_files, attribute_names):
        print(f'Extracting data from {attribute_file}...')

        # Open the attribute file
        with rasterio.open(attribute_file) as attribute_src:
            # Read the raster data within the polygon using the mask
            values = attribute_src.read(1, masked=True)

            # Iterate through each polygon in the GeoPackage
            for index, row in gdf.iterrows():
                # Get the geometry of the polygon
                geom = row['geometry']

                # Use geometry mask to create a mask for the polygon
                mask = geometry_mask([geom], out_shape=attribute_src.shape,
                                     transform=attribute_src.transform, invert=True)

                masked_values = np.ma.array(values, mask=~mask)

                # Find the mean value within the polygon
                max_val = np.max(masked_values)

                # Save the mean value to the GeoDataFrame
                gdf.at[index, attribute_name] = max_val

    # Save the updated GeoDataFrame back to the GeoPackage
    gdf.to_file(f'{config.output_dir}/buildings_with_attributes.gpkg',
                layer='buildings', driver='GPKG')

    print('Done.')


if __name__ == '__main__':
    main()
