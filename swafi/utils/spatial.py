import pandas as pd
import geopandas as gpd
import math
import numpy as np
import rasterio
from rasterio import features


def extract_statistics(domain, data_path, categorical=False, categories=None):
    """
    Extract statistics from a raster file for each cell in the domain.
    The statistics are: min, max, mean, std, median.

    Parameters
    ----------
    domain : Domain
        The domain object.
    data_path : str|Path
        The path to the raster file.
    categorical : bool
        Whether the data is categorical or not.
    categories : list
        The list of categories if the data is categorical.

    Returns
    -------
    df : DataFrame
        A DataFrame containing the statistics for each cell.
    """
    # Open the data files using rasterio
    with rasterio.open(data_path) as data_ds:
        # Initialize empty lists to store statistics for each cell
        cids_vals = []
        if categorical:
            if categories is None:
                raise ValueError("Categories must be provided when the "
                                 "data is categorical.")
            cat_vals = [None] * len(categories)
            for i_cat, category in enumerate(categories):
                cat_vals[i_cat] = []
        else:
            min_vals = []
            max_vals = []
            mean_vals = []
            std_vals = []
            median_vals = []

        # Iterate over each cell
        cids_map = domain.cids['ids_map']
        for i in range(cids_map.shape[0]):
            for j in range(cids_map.shape[1]):
                cid = cids_map[i, j]
                if math.isnan(cid) or cid == 0:
                    continue

                # Calculate the bounding box of the corresponding cell in the data
                x = domain.cids['xs'][i, j]
                y = domain.cids['ys'][i, j]
                res = domain.resolution[0]
                bbox = rasterio.coords.BoundingBox(
                    x - res / 2, y - res / 2, x + res / 2, y + res / 2
                )

                # Read the data within the bounding box
                data_cells = data_ds.read(1, window=data_ds.window(*bbox))
                data_cells[data_cells == data_ds.nodata] = np.nan
                data_cells[data_cells == -9999] = np.nan

                # Skip empty cells
                if data_cells.shape[0] == 0 or data_cells.shape[1] == 0:
                    continue
                if np.isnan(data_cells).all():
                    continue

                # Calculate statistics and append them to the lists
                cids_vals.append(int(cid))
                if categorical:
                    for i_cat, category in enumerate(categories):
                        nb_cells_cat = np.nansum(data_cells == category)
                        nb_cells_other = np.nansum(data_cells != category)
                        ratio = nb_cells_cat / (nb_cells_cat + nb_cells_other)
                        cat_vals[i_cat].append(ratio)
                else:
                    min_vals.append(np.nanmin(data_cells))
                    max_vals.append(np.nanmax(data_cells))
                    mean_vals.append(np.nanmean(data_cells))
                    std_vals.append(np.nanstd(data_cells))
                    median_vals.append(np.nanmedian(data_cells))

    # Create a pandas DataFrame to store the statistics
    tag = data_path.stem
    if categorical:
        df_dict = {'cid': cids_vals}
        for i_cat, category in enumerate(categories):
            df_dict[f'{tag}_cat_{category}'] = cat_vals[i_cat]
        df = pd.DataFrame(df_dict)
    else:
        df = pd.DataFrame({
            'cid': cids_vals,
            f'{tag}_min': min_vals,
            f'{tag}_max': max_vals,
            f'{tag}_mean': mean_vals,
            f'{tag}_std': std_vals,
            f'{tag}_median': median_vals,
        })

    return df


def rasterize(vector_path, ref_raster_path):
    """
    Rasterize a vector according to the properties of the provided raster file.
    Heavily based on https://pygis.io/docs/e_raster_rasterize.html

    Parameters
    ----------
    vector_path: str
        Path of the vector file to rasterize.
    ref_raster_path: str
        Path of the raster to use as reference

    Returns
    -------
    The rasterized vector.
    """
    vector = gpd.read_file(vector_path)
    geoms = [shapes for shapes in vector.geometry]

    ref_raster = rasterio.open(ref_raster_path)

    rasterized = features.rasterize(
        geoms,
        out_shape=ref_raster.shape,
        fill=0,
        out=None,
        transform=ref_raster.transform,
        all_touched=True,
        default_value=1,
        dtype=float)

    return rasterized
