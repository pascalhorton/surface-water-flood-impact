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


class DamagesGvz(Damages):
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
            'a',  # most likely surface flood
            'b',
            'c',
            'd',
            'e']  # most likely fluvial flood

        self.selected_categories = [
            'a',
            'b']

        self.exposure = pd.DataFrame(
            columns=['year', 'mask_index', 'selection'] + self.categories)
        self.claims = pd.DataFrame(
            columns=['date_claim', 'mask_index', 'selection'])

        self.exposure = self.exposure.astype('int32')
        self.claims = self.claims.astype('int32')
        self.claims['date_claim'] = pd.to_datetime(self.claims['date_claim'])

        self._create_exposure_claims_df()
        self._load_from_dump('damages_gvz')

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
            The types of categories to get. Can be 'most_likely_surface_flood',
            'likely_surface_flood', 'possibly_surface_flood', 'likely_fluvial_flood',
            'most_likely_fluvial_flood'.

        Returns
        -------
        The list of corresponding categories.
        """
        columns = self.categories

        if isinstance(types, str):
            types = [types]

        for cat_type in types:
            if cat_type.lower() == 'most_likely_surface_flood':
                columns = [i for i in columns if i in ['a']]
                continue
            if cat_type.lower() == 'likely_surface_flood':
                columns = [i for i in columns if i in ['a', 'b']]
                continue
            if cat_type.lower() == 'possibly_surface_flood':
                columns = [i for i in columns if i in ['a', 'b', 'c']]
                continue
            if cat_type.lower() == 'likely_fluvial_flood':
                columns = [i for i in columns if i in ['d', 'e']]
                continue
            if cat_type.lower() == 'most_likely_fluvial_flood':
                columns = [i for i in columns if i in ['e']]
                continue

        return columns

    def _extract_exposure_data(self, directory):
        """
        Extract all contract data.
        """
        pass

    def _parse_exposure_files(self, files):
        """
        Parse the provided exposure files.
        """
        pass

    def _extract_claim_data(self, directory):
        """
        Extracts all claims data.
        """
        pass

    def _parse_claim_files(self, files, category):
        """
        Parse the claim files for a given category.
        """
        pass

    @staticmethod
    def _extract_date_from_filename(file):
        """
        Extracts the date from the file name.
        """
        pass
