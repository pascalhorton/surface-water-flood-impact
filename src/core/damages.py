"""
Class to handle all contracts and claims.
"""

import glob
import pickle
import ntpath
from datetime import datetime
from pathlib import Path

import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.config import Config

config = Config()


class Damages:
    def __init__(self, cid_file, year_start=None, year_end=None, use_dump=True,
                 dataset='mobi_2023'):
        """
        The Damages class.

        Parameters
        ----------
        cid_file: str
            Path to the CID file containing the IDs of the cells
        year_start: int
            The starting year of the data.
        year_end: int
            The ending year of the data.
        use_dump: bool
            Dump the content to the TMP_DIR and load if available
        dataset: str
            The dataset ID (default 'mobi_2023')
        """
        self.use_dump = use_dump

        self.crs = config.get('CRS', 'EPSG:2056')
        self.shape = None
        self.extent = None
        self.mask = None
        self.cids = dict(extent=None, ids=None)

        self.year_start = year_start
        if not self.year_start:
            self.year_start = config.get('YEAR_START', 2013)
        self.year_end = year_end
        if not self.year_end:
            self.year_end = config.get('YEAR_END', 2022)

        self.categories = ['claims']
        self.tags_contracts = ['*']
        self.tags_claims = ['*']

        if dataset == 'mobi_2023':
            self.categories = ['sme_ext_cont',  # SME, external, content
                               'sme_ext_struc',  # SME, external, structure
                               'sme_int_cont',  # SME, internal, content
                               'sme_int_struc',  # SME, internal, structure
                               'priv_ext_cont',  # Private, external, content
                               'priv_ext_struc',  # Private, external, structure
                               'priv_int_cont',  # Private, internal, content
                               'priv_int_struc']  # Private, internal, structure

            self.tags_contracts = ['KMU_ES_FH',
                                   'KMU_ES_GB',
                                   'KMU_W_FH',
                                   'KMU_W_GB',
                                   'Privat_ES_FH',
                                   'Privat_ES_GB',
                                   'Privat_W_FH',
                                   'Privat_W_GB']

            self.tags_claims = ['Ueberschwemmung_KMU_FH',
                                'Ueberschwemmung_KMU_GB',
                                'Wasser_KMU_FH',
                                'Wasser_KMU_GB',
                                'Ueberschwemmung_Privat_FH',
                                'Ueberschwemmung_Privat_GB',
                                'Wasser_Privat_FH',
                                'Wasser_Privat_GB']

        self.contracts = pd.DataFrame(
            columns=['year', 'index', 'selection'] + self.categories)
        self.claims = pd.DataFrame(
            columns=['date_claim', 'index', 'selection'])

        self.contracts = self.contracts.astype('int32')
        self.claims = self.claims.astype('int32')
        self.claims['date_claim'] = pd.to_datetime(self.claims['date_claim'])

        self._load_from_dump()
        self._load_cid_file(cid_file)

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

        contract_data = self._extract_contract_data(directory)
        self._create_mask(contract_data)

        for idx, contracts in enumerate(contract_data):
            contract_data_cat = self._extract_data_with_mask(contracts)
            if idx == 0:
                self._initialize_contracts_dataframe(contract_data_cat)
            self._set_to_contracts_dataframe(contract_data_cat, self.categories[idx])

        self._dump_object()

    def load_claims(self, directory=None):
        """
        Load the claim data from geotiff files.

        Parameters
        ----------
        directory: str
            The path to the directory containing the files.
        """
        if self.use_dump and not self.claims.empty:
            print("Claims reloaded from pickle file.")
            return

        if not directory:
            directory = config.get('DIR_CLAIMS')

        self._extract_claim_data(directory)
        self._clean_claims_dataframe()

        self._dump_object()

    def select_all_categories(self):
        columns = self.categories
        self.contracts['selection'] = self.contracts[columns].sum(axis=1)
        self.claims['selection'] = self.claims[columns].sum(axis=1)

    def _extract_contract_data(self, directory):
        """
        Extract all contract data.
        """
        contracts = glob.glob(directory + '/*.tif')
        contract_data = []
        for idx in tqdm(range(len(self.tags_contracts)), desc="Extracting contracts"):
            tag = self.tags_contracts[idx]
            files = [s for s in contracts if tag in s]
            data = self._parse_contract_files(files)
            contract_data.append(data)

        return contract_data

    def _parse_contract_files(self, files):
        """
        Parse the provided contract files.
        """
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

    def _extract_claim_data(self, directory):
        """
        Extracts all claims data.
        """
        claims = glob.glob(directory + '/*.tif')
        for idx, tag in enumerate(self.tags_claims):
            tag = self.tags_claims[idx]
            files = [s for s in claims if tag in s]
            files.sort()
            self._parse_claim_files(files, self.categories[idx])

    def _parse_claim_files(self, files, category):
        """
        Parse the claim files for a given category.
        """
        df_claims = pd.DataFrame(columns=['date_claim', 'index', category])
        df_claims = df_claims.astype('int32')
        df_claims['date_claim'] = pd.to_datetime(df_claims['date_claim'])

        for i_file in tqdm(range(len(files)), desc=f"Extracting {category}"):
            file = files[i_file]
            with rasterio.open(file) as dataset:
                self._check_projection(dataset, file)
                self._check_extent(dataset, file)
                data = dataset.read()
                self._check_shape(data, file)

                # Claims can be empty for a given type of object or phenomenon
                if data.sum() == 0:
                    continue

                indices, values = self._extract_non_null_claims(data)
                date = self._extract_date_from_filename(file)
                df_event = pd.DataFrame(columns=['date_claim', 'index', category])
                df_event['date_claim'] = [date] * len(indices)
                df_event['index'] = indices
                df_event[category] = values
                df_claims = pd.concat([df_claims, df_event])

        self._store_in_claims_dataframe(df_claims)

    def _store_in_claims_dataframe(self, df_claims):
        """
        Stores the claims for a given category in the dataframe.
        """
        self.claims = pd.merge(self.claims, df_claims, how="outer",
                               on=["date_claim", "index"], validate="one_to_one")

    def _extract_non_null_claims(self, data):
        """
        Extracts the cells with at least 1 claim.
        """
        # Extract the pixels where the catalog is not null
        extracted = np.extract(self.mask, data[0, :, :])
        if data.sum() != extracted.sum():
            raise RuntimeError(
                f"Missed claims during extraction: {data.sum() - extracted.sum()}")

        # Get non null data
        indices = np.nonzero(extracted)[0]
        values = extracted[indices]
        return indices, values

    @staticmethod
    def _extract_date_from_filename(file):
        """
        Extracts the date from the file name.
        """
        filename = ntpath.basename(file)
        filename_date = filename.split("_")[1]
        date = datetime.strptime(filename_date, '%Y%m%d').date()
        return date

    def _check_projection(self, dataset, file):
        """
        Check projection consistency with other files.
        """
        if dataset.crs != self.crs:
            raise RuntimeError(
                f"The projection of {file} differs from the project one.")

    def _check_extent(self, dataset, file):
        """
        Check extent consistency with other files.
        """
        if not self.extent:
            self.extent = dataset.bounds
        elif self.extent != dataset.bounds:
            raise RuntimeError(f"The extent of {file} differs from other files.")

    def _check_shape(self, data, file):
        """
        Check shape consistency with other files.
        """
        if not self.shape:
            self.shape = data.shape
        elif self.shape != data.shape:
            raise RuntimeError(f"The shape of {file} differs from other files.")

    def _create_mask(self, contract_data):
        """
        Creates a mask with True for all pixels containing at least 1 annual contract.
        """
        self.mask = np.zeros(self.shape[1:], dtype=bool)
        for arr in contract_data:
            max_value = arr.max(axis=0)
            self.mask[max_value > 0] = True

    def _extract_data_with_mask(self, data):
        """
        Extracts data according to the mask and returns a 1-D array.
        """
        if self.mask is None:
            raise RuntimeError("The mask for extraction was not defined.")
        extracted = np.zeros((data.shape[0], np.sum(self.mask)), dtype=np.int16)
        for i in range(data.shape[0]):
            extracted[i, :] = np.extract(self.mask, data[i, :, :])
        return extracted

    def _load_from_dump(self):
        """
        Loads the object content from a pickle file.
        """
        if not self.use_dump:
            return
        tmp_dir = config.get('TMP_DIR')
        file_path = Path(tmp_dir + '/damages.pickle')
        if file_path.is_file():
            with open(file_path, 'rb') as f:
                values = pickle.load(f)
                self.shape = values.shape
                self.extent = values.extent
                self.mask = values.mask
                self.cids = values.cids
                self.contracts = values.contracts
                self.claims = values.claims

    def _dump_object(self):
        """
        Saves the object content to a pickle file.
        """
        if not self.use_dump:
            return
        tmp_dir = config.get('TMP_DIR')
        file_path = Path(tmp_dir + '/damages.pickle')
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def _initialize_contracts_dataframe(self, contract_data_cat):
        """
        Initializes the contracts dataframe by filling the year and the index columns.
        The index column refers to the 1-D array after extraction by the mask.
        """
        n_years = self.year_end - self.year_start + 1
        n_annual_rows = contract_data_cat.shape[1]
        years = np.repeat(np.arange(self.year_start, self.year_end + 1), n_annual_rows)
        self.contracts['year'] = years
        indices = np.tile(np.arange(n_annual_rows), n_years)
        self.contracts['index'] = indices

    def _set_to_contracts_dataframe(self, contract_data_cat, category):
        """
        Sets the contract data to the dataframe for the given category.
        """
        contracts = np.reshape(contract_data_cat, contract_data_cat.size)
        self.contracts[category] = contracts

    def _clean_claims_dataframe(self):
        """
        Reorder claims dataframe and remove nans.
        """
        columns = ['date_claim', 'index', 'selection'] + self.categories
        self.claims = self.claims.reindex(columns=columns)
        self.claims.fillna(0, inplace=True)
        self.claims.sort_values(by=['date_claim', 'index'], inplace=True)
        self.claims.reset_index(inplace=True, drop=True)
        for category in self.categories:
            self.claims[category] = self.claims[category].astype('int32')

    def _load_cid_file(self, cid_file):
        if self.use_dump and self.cids['extent'] is not None:
            print("CIDs reloaded from pickle file.")
            return
        with rasterio.open(cid_file) as dataset:
            self._check_projection(dataset, cid_file)
            data = np.nan_to_num(dataset.read())
            data = data.astype(np.int32)
            self.cids['ids'] = data
            self.cids['extent'] = dataset.bounds
