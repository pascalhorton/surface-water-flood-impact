"""
Class to handle all exposure and claims.
"""

import glob
import ntpath
from datetime import datetime

import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm

from .damages import Damages
from .config import Config

config = Config()


class DamagesMobiliar(Damages):
    def __init__(self, cid_file=None, year_start=None, year_end=None, use_dump=True,
                 dir_exposure=None, dir_claims=None, pickle_file=None, pickle_dir=None):
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
            Dump the content to the PICKLES_DIR and load if available
        dir_exposure: str
            The path to the directory containing the exposure/contract files.
        dir_claims: str
            The path to the directory containing the claim files.
        pickle_file: str
            The path to a pickle file to load.
        pickle_dir: str
            The path to the working directory for pickle files
        """
        super().__init__(cid_file=cid_file, year_start=year_start, year_end=year_end,
                         use_dump=use_dump, pickle_dir=pickle_dir)

        self.categories = [
            'sme_ext_cont_pluv',  # SME, external, content, pluvial
            'sme_ext_cont_fluv',  # SME, external, content, fluvial
            'sme_ext_struc_pluv',  # SME, external, structure, pluvial
            'sme_ext_struc_fluv',  # SME, external, structure, fluvial
            'sme_int_cont',  # SME, internal, content
            'sme_int_struc',  # SME, internal, structure
            'priv_ext_cont_pluv',  # Private, external, content, pluvial
            'priv_ext_cont_fluv',  # Private, external, content, fluvial
            'priv_ext_struc_pluv',  # Private, external, structure, pluvial
            'priv_ext_struc_fluv',  # Private, external, structure, fluvial
            'priv_int_cont',  # Private, internal, content
            'priv_int_struc']  # Private, internal, structure

        self.selected_categories = [
            'sme_ext_cont_pluv',
            'sme_ext_struc_pluv',
            'priv_ext_cont_pluv',
            'priv_ext_struc_pluv']

        self.tags_exposure = [
            'KMU_ES_FH',
            'KMU_ES_FH',
            'KMU_ES_GB',
            'KMU_ES_GB',
            'KMU_W_FH',
            'KMU_W_GB',
            'Privat_ES_FH',
            'Privat_ES_FH',
            'Privat_ES_GB',
            'Privat_ES_GB',
            'Privat_W_FH',
            'Privat_W_GB']

        self.tags_claims = [
            'Ueberschwemmung_pluvial_KMU_FH',
            'Ueberschwemmung_fluvial_KMU_FH',
            'Ueberschwemmung_pluvial_KMU_GB',
            'Ueberschwemmung_fluvial_KMU_GB',
            'Wasser_KMU_FH',
            'Wasser_KMU_GB',
            'Ueberschwemmung_pluvial_Privat_FH',
            'Ueberschwemmung_fluvial_Privat_FH',
            'Ueberschwemmung_pluvial_Privat_GB',
            'Ueberschwemmung_fluvial_Privat_GB',
            'Wasser_Privat_FH',
            'Wasser_Privat_GB']

        self._create_exposure_claims_df()
        self._load_from_dump('damages_mobiliar')

        if dir_exposure is not None:
            self.load_exposure(dir_exposure)

        if dir_claims is not None:
            self.load_claims(dir_claims)

        if pickle_file is not None:
            self.load_from_pickle(pickle_file)

    def get_categories_from_type(self, types):
        """
        Get the categories from types.

        Parameters
        ----------
        types: str or list
            The types of categories to get. Can be 'external', 'internal', 'sme',
            'private', 'content', 'structure'.

        Returns
        -------
        The list of corresponding categories.
        """
        columns = self.categories

        if isinstance(types, str):
            types = [types]

        for cat_type in types:
            if cat_type.lower() == 'external':
                columns = [i for i in columns if 'ext' in i]
                continue
            if cat_type.lower() == 'internal':
                columns = [i for i in columns if 'int' in i]
                continue
            if cat_type.lower() == 'sme':
                columns = [i for i in columns if 'sme' in i]
                continue
            if cat_type.lower() == 'private':
                columns = [i for i in columns if 'priv' in i]
                continue
            if cat_type.lower() == 'content':
                columns = [i for i in columns if 'cont' in i]
                continue
            if cat_type.lower() == 'structure':
                columns = [i for i in columns if 'struc' in i]
                continue
            if cat_type.lower() == 'pluvial':
                columns = [i for i in columns if 'pluv' in i]

        return columns

    def _extract_exposure_data(self, directory):
        """
        Extract all contract data.
        """
        contracts = glob.glob(directory + '/*.tif')
        exposure_data = []
        for idx in tqdm(range(len(self.tags_exposure)), desc="Extracting exposure"):
            tag = self.tags_exposure[idx]
            files = [s for s in contracts if tag in s]
            data = self._parse_exposure_files(files)
            exposure_data.append(data)

        return exposure_data

    def _parse_exposure_files(self, files):
        """
        Parse the provided exposure files.
        """
        all_data = None
        for year in range(self.year_start, self.year_end + 1):
            file = [s for s in files if f'_{year}' in s]
            if len(file) != 1:
                raise RuntimeError(f"{len(file)} files found instead of 1.")
            file = file[0]
            with rasterio.open(file) as dataset:
                self.domain.check_projection(dataset, file)
                self.domain.check_resolution(dataset, file)
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
        df_claims = pd.DataFrame(columns=['date_claim', 'mask_index', category])
        df_claims = df_claims.astype('int32')
        df_claims['date_claim'] = pd.to_datetime(df_claims['date_claim'])

        for i_file in tqdm(range(len(files)), desc=f"Extracting {category}"):
            file = files[i_file]
            with rasterio.open(file) as dataset:
                self.domain.check_projection(dataset, file)
                self.domain.check_resolution(dataset, file)
                self._check_extent(dataset, file)
                data = dataset.read()
                self._check_shape(data, file)

                # Claims can be empty for a given type of object or phenomenon
                if data.sum() == 0:
                    continue

                indices, values = self._extract_non_null_claims(data)
                date = self._extract_date_from_filename(file)
                df_case = pd.DataFrame(columns=['date_claim', 'mask_index', category])
                df_case['date_claim'] = [date] * len(indices)
                df_case['mask_index'] = indices
                df_case[category] = values
                df_claims = pd.concat([df_claims, df_case])

        self._store_in_claims_dataframe(df_claims)

    @staticmethod
    def _extract_date_from_filename(file):
        """
        Extracts the date from the file name.
        """
        filename = ntpath.basename(file)
        filename_date = filename.split("_")[1]
        return datetime.strptime(filename_date, '%Y%m%d').date()
