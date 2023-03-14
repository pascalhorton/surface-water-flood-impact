"""
Extract the annual contracts.
"""

import glob
import pickle
from pathlib import Path

import rasterio
import numpy as np

from utils.config import Config

config = Config()


class Targets:
    def __init__(self, year_start=None, year_end=None, use_dump=True):
        """
        The Targets class.

        Parameters
        ----------
        year_start: int
            The starting year of the data.
        year_end: int
            The ending year of the data.
        use_dump: bool
            Dump the content to the TMP_DIR and load if available
        """
        self.use_dump = use_dump

        self.crs = config.get('CRS', 'EPSG:2056')
        self.shape = None
        self.extent = None
        self.mask = None

        self.year_start = year_start
        if not self.year_start:
            self.year_start = config.get('YEAR_START', 2013)
        self.year_end = year_end
        if not self.year_end:
            self.year_end = config.get('YEAR_END', 2022)

        self.tags = ['KMU_ES_FH',
                     'KMU_ES_GB',
                     'KMU_W_FH',
                     'KMU_W_GB',
                     'Privat_ES_FH',
                     'Privat_ES_GB',
                     'Privat_W_FH',
                     'Privat_W_GB']

        # Private (Privat) vs SME (KMU)
        self.private = [False, False, False, False, True, True, True, True]
        self.sme = [True, True, True, True, False, False, False, False]
        # External (ES) vs internal (W)
        self.external = [True, True, False, False, True, True, False, False]
        self.internal = [False, False, True, True, False, False, True, True]
        # Content (FH) vs structural (GB)
        self.content = [True, False, True, False, True, False, True, False]
        self.structural = [False, True, False, True, False, True, False, True]

        self.contracts = [np.zeros((1, 1, 1))] * len(self.tags)

        self._load_from_dump()

    def load_contracts(self, directory=None):
        """
        Load the contract data from geotiff files.

        Parameters
        ----------
        directory: str
            The path to the directory containing the files.
        """
        if self.use_dump and self.shape is not None:
            print("Contracts reloaded from pickle file.")
            return

        if not directory:
            directory = config.get('DIR_CONTRACTS')

        self._extract_contract_data(directory)
        self._create_mask()

        for idx, contracts in enumerate(self.contracts):
            self.contracts[idx] = self._compress_data(contracts)

        self._dump_object()

    def load_damages(self, directory=None):
        """
        Load the damage data from geotiff files.

        Parameters
        ----------
        directory: str
            The path to the directory containing the files.
        """
        if not directory:
            directory = config.get('DIR_DAMAGES')

        self._dump_object()

    def _extract_contract_data(self, directory):
        contracts = glob.glob(directory + '/*.tif')
        for idx, tag in enumerate(self.tags):
            files = [s for s in contracts if tag in s]
            self.contracts[idx] = self._parse_contract_files(files)

    def _parse_contract_files(self, files):
        all_data = None
        for year in range(self.year_start, self.year_end + 1):
            file = [s for s in files if f'_{year}' in s]
            if len(file) != 1:
                raise RuntimeError(f"{len(file)} files found instead of 1.")
            file = file[0]
            with rasterio.open(file) as dataset:
                self._check_projection(dataset, file)
                self._check_extent(dataset, file)
                data = dataset.read()
                self._check_shape(data, file)
                if all_data is None:
                    all_data = data
                else:
                    all_data = np.append(all_data, data, axis=0)
        return all_data

    def _check_projection(self, dataset, file):
        if dataset.crs != self.crs:
            raise RuntimeError(
                f"The projection of {file} differs from the project one.")

    def _check_extent(self, dataset, file):
        if not self.extent:
            self.extent = dataset.bounds
        elif self.extent != dataset.bounds:
            raise RuntimeError(f"The extent of {file} differs from other files.")

    def _check_shape(self, data, file):
        if not self.shape:
            self.shape = data.shape
        elif self.shape != data.shape:
            raise RuntimeError(f"The shape of {file} differs from other files.")

    def _create_mask(self):
        self.mask = np.zeros(self.shape[1:], dtype=bool)
        for arr in self.contracts:
            max_value = arr.max(axis=0)
            self.mask[max_value > 0] = True

    def _compress_data(self, data):
        if self.mask is None:
            raise RuntimeError("The mask for extraction was not defined.")
        extracted = np.zeros((data.shape[0], np.sum(self.mask)))
        for i in range(data.shape[0]):
            extracted[i, :] = np.extract(self.mask, data[i, :, :])
        return extracted

    def _load_from_dump(self):
        if not self.use_dump:
            return
        tmp_dir = config.get('TMP_DIR')
        file_path = Path(tmp_dir + '/targets.pickle')
        if file_path.is_file():
            with open(file_path, 'rb') as f:
                values = pickle.load(f)
                self.shape = values.shape
                self.extent = values.extent
                self.mask = values.mask
                self.contracts = values.contracts

    def _dump_object(self):
        if not self.use_dump:
            return
        tmp_dir = config.get('TMP_DIR')
        file_path = Path(tmp_dir + '/targets.pickle')
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
