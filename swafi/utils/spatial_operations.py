import geopandas as gpd
import rasterio
from rasterio import features


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

    rasterized = features.rasterize(geoms,
                                    out_shape=ref_raster.shape,
                                    fill=0,
                                    out=None,
                                    transform=ref_raster.transform,
                                    all_touched=True,
                                    default_value=1,
                                    dtype=float)

    return rasterized
