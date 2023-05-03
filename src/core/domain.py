"""
Class to define the spatial domain and cell IDs.
"""

import pickle
import rasterio
import numpy as np
from pathlib import Path

from utils.config import Config

config = Config()


class Domain:
    def __init__(self, cid_file=None):
        """
        The Domain class. Defines the cell IDs.
        """
        self.crs = config.get('CRS', 'EPSG:2056')
        self.resolution = None
        self.cids = dict(extent=None, ids_map=np.array([]), ids_list=np.array([]),
                         xs=np.array([]), ys=np.array([]))

        if not cid_file:
            cid_file = config.get('CID_PATH')

        self._load_from_dump()
        self._load_cid_file(cid_file)

    def check_projection(self, dataset, file):
        """
        Check projection consistency with other files.

        Parameters
        ----------
        dataset: rasterio dataset
            The dataset to test.
        file: str
            The file name.
        """
        if dataset.crs != self.crs:
            raise RuntimeError(
                f"The projection of {file} differs from the project one.")

    def check_resolution(self, dataset, file):
        """
        Check the resolution consistency with other files.

        Parameters
        ----------
        dataset: rasterio dataset
            The dataset to test.
        """
        if self.resolution is None:
            self.resolution = dataset.res
        if dataset.res != self.resolution:
            raise RuntimeError(
                f"The resolution of {file} differs from the project one.")

    def create_cids_list(self, xs_mask, ys_mask):
        """
        Create the list of CIDs.

        Parameters
        ----------
        xs_mask: list
            X coordinates from the mask.
        ys_mask: list
            Y coordinates from the mask.
        """
        cids = np.zeros(len(xs_mask))
        xs_cid = self.cids['xs'][0, :]
        ys_cid = self.cids['ys'][:, 0]

        for i, (x, y) in enumerate(zip(xs_mask, ys_mask)):
            cids[i] = self.cids['ids_map'][ys_cid == y, xs_cid == x]

        self.cids['ids_list'] = cids

    def _load_cid_file(self, cid_file):
        """
        Load the file containing the CIDs.
        """
        if self.cids['extent'] is not None:
            return
        with rasterio.open(cid_file) as dataset:
            self.check_projection(dataset, cid_file)
            self.check_resolution(dataset, cid_file)
            data = np.nan_to_num(dataset.read())
            data = data.astype(np.int32)
            data = data.squeeze(axis=0)
            self.cids['ids_map'] = data
            self.cids['extent'] = dataset.bounds

            # Extract the axes
            cols, rows = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
            xs, ys = rasterio.transform.xy(dataset.transform, rows, cols)
            self.cids['xs'] = np.array(xs)
            self.cids['ys'] = np.array(ys)

        self._dump_object()

    def _load_from_dump(self, filename='damages.pickle'):
        """
        Loads the object content from a pickle file.
        """
        tmp_dir = config.get('TMP_DIR')
        file_path = Path(tmp_dir + '/' + filename)
        if file_path.is_file():
            with open(file_path, 'rb') as f:
                values = pickle.load(f)
                self.resolution = values.resolution
                self.cids = values.cids

    def _dump_object(self, filename='damages.pickle'):
        """
        Saves the object content to a pickle file.
        """
        tmp_dir = config.get('TMP_DIR')
        file_path = Path(tmp_dir + '/' + filename)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)