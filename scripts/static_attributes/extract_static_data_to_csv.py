"""
Extract static data from geotiff files based on the cids and save the results as csv.
"""

import rasterio
import math
import numpy as np
import pandas as pd
from pathlib import Path
from swafi.config import Config
from swafi.domain import Domain

# Select the attributes of interest. Options are: static_terrain,
# static_flowacc_pysheds, static_flowacc_richdem
SRC_DIR = 'static_terrain'

config = Config(output_dir='static_attributes_csv')
base_dir = config.get('OUTPUT_DIR')
data_dir = Path(base_dir + '/' + SRC_DIR)


def main():
    domain = Domain(config.get('CID_PATH'))
    cid_file = config.get('CID_PATH')

    # List all .tif files in the source directory
    tif_files = [f for f in data_dir.glob('*.tif')]

    # Extract the static data for each file
    df = pd.DataFrame()
    for f in tif_files:
        print(f'Extracting static data from {f.name}...')

        with rasterio.open(f) as dataset:
            domain.check_projection(dataset, f)

        df_new = extract_statistics(cid_file, f)

        # Merge the statistics to the DataFrame by the column cid
        if df.empty:
            df = df_new
        else:
            df = df.merge(df_new, on='cid', how='outer')

    # Save the DataFrame as csv
    df.to_csv(f'{config.output_dir}/{SRC_DIR}.csv', index=False)

    print('Done.')


def extract_statistics(cids_path, data_path):
    # Open the cids.tif and data files using rasterio
    with rasterio.open(cids_path) as cids_ds, rasterio.open(data_path) as data_ds:
        # Read the raster data as numpy arrays
        cids_data = cids_ds.read(1)

        # Initialize empty lists to store statistics for each cell in cids_data
        cids_vals = []
        min_vals = []
        max_vals = []
        mean_vals = []
        std_vals = []
        median_vals = []

        # Iterate over each cell in cids_data
        for i in range(cids_data.shape[0]):
            for j in range(cids_data.shape[1]):
                if math.isnan(cids_data[i, j]):
                    continue

                # Calculate the bounding box of the corresponding cell in the data
                x, y = cids_ds.xy(j, i)
                res = cids_ds.res[0]
                bbox = rasterio.coords.BoundingBox(
                    x - res / 2, y - res / 2, x + res / 2, y + res / 2
                )

                # Read the data within the bounding box
                window = data_ds.window(*bbox)
                data_cells = data_ds.read(1, window=data_ds.window(*bbox))

                # Skip empty cells
                if data_cells.shape[0] == 0 or data_cells.shape[1] == 0:
                    continue
                if np.isnan(data_cells).all():
                    continue
                if np.any(data_cells == -9999):
                    continue

                # Calculate statistics and append them to the lists
                cids_vals.append(int(cids_data[i, j]))
                min_vals.append(np.nanmin(data_cells))
                max_vals.append(np.nanmax(data_cells))
                mean_vals.append(np.nanmean(data_cells))
                std_vals.append(np.nanstd(data_cells))
                median_vals.append(np.nanmedian(data_cells))

    # Create a pandas DataFrame to store the statistics
    tag = data_path.stem
    df = pd.DataFrame({
        'cid': cids_vals,
        f'{tag}_min': min_vals,
        f'{tag}_max': max_vals,
        f'{tag}_mean': mean_vals,
        f'{tag}_std': std_vals,
        f'{tag}_median': median_vals,
    })

    return df


if __name__ == '__main__':
    main()
