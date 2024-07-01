"""
Extract static data from geotiff files based on the cids and save the results as csv.
"""

import rasterio
import pandas as pd
from pathlib import Path
from swafi.config import Config
from swafi.domain import Domain
from swafi.utils.spatial import extract_statistics

# Select the attributes of interest. Options are:
# static_terrain,
# static_flowacc_pysheds,
# static_flowacc_richdem,
# static_runoff_coeff,
# static_land_cover,
# static_twi,
SRC_DIR = 'static_twi'
CATEGORICAL = False
CATEGORIES = range(1, 13)

config = Config(output_dir='static_attributes_csv')
base_dir = config.get('OUTPUT_DIR')
data_dir = Path(base_dir + '/' + SRC_DIR)


def main():
    domain = Domain()

    # List all .tif files in the source directory
    tif_files = [f for f in data_dir.glob('*.tif')]

    # Extract the static data for each file
    df = pd.DataFrame()
    for f in tif_files:
        print(f'Extracting static data from {f.name}...')

        with rasterio.open(f) as dataset:
            domain.check_projection(dataset, f)

        if CATEGORICAL:
            df_new = extract_statistics(domain, f, categorical=True,
                                        categories=CATEGORIES)
        else:
            df_new = extract_statistics(domain, f)

        # Merge the statistics to the DataFrame by the column cid
        if df.empty:
            df = df_new
        else:
            df = df.merge(df_new, on='cid', how='outer')

    # Save the DataFrame as csv
    df.to_csv(f'{config.output_dir}/{SRC_DIR}.csv', index=False)

    print('Done.')


if __name__ == '__main__':
    main()
