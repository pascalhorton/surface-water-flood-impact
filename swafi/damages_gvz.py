"""
Class to handle all exposure and claims.
"""

import glob
import ntpath
from datetime import datetime

import rasterio
import numpy as np
import pandas as pd
import netCDF4 as nc4
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
            The name of a pickle file to load.
        pickle_dir: str
            The path to the working directory for pickle files
        """
        super().__init__(cid_file=cid_file, year_start=year_start, year_end=year_end,
                         use_dump=use_dump, pickle_dir=pickle_dir)

        self.name = 'gvz'

        self.exposure_categories = [
            'all_buildings']

        self.claim_categories = [
            'A',  # most likely surface flood
            'B',
            'C',
            'D',
            'E']  # most likely fluvial flood

        self._create_exposure_claims_df()
        self._load_from_dump('damages_gvz.pickle')

        if dir_exposure is not None:
            self.load_exposure(dir_exposure)

        if dir_claims is not None:
            self.load_claims(dir_claims)

        if pickle_file is not None:
            self.load_from_pickle(pickle_file)

        self._remove_data_outside_period()

    def get_claim_categories_from_type(self, types):
        """
        Get the claim categories from types.

        Parameters
        ----------
        types: str or list
            The types of claim categories to get. Can be 'most_likely_pluvial',
            'likely_pluvial', 'fluvial_or_pluvial', 'likely_fluvial',
            'most_likely_fluvial'.

        Returns
        -------
        The list of corresponding claim categories.
        """
        columns = self.claim_categories

        if isinstance(types, str):
            types = [types]

        for cat_type in types:
            if cat_type.lower() == 'most_likely_pluvial':
                columns = [i for i in columns if i in ['A']]
                continue
            elif cat_type.lower() == 'likely_pluvial':
                columns = [i for i in columns if i in ['A', 'B']]
                continue
            elif cat_type.lower() == 'fluvial_or_pluvial':
                columns = [i for i in columns if i in ['A', 'B', 'C', 'D', 'E']]
                continue
            elif cat_type.lower() == 'likely_fluvial':
                columns = [i for i in columns if i in ['D', 'E']]
                continue
            elif cat_type.lower() == 'most_likely_fluvial':
                columns = [i for i in columns if i in ['E']]
                continue
            elif cat_type in ['A', 'B', 'C', 'D', 'E']:
                columns = [i for i in columns if i in [cat_type]]
                continue
            else:
                raise ValueError(f"Unknown claim type: {cat_type}")

        return columns

    def get_exposure_categories_from_type(self, types):
        """
        Get the exposure categories from types.

        Parameters
        ----------
        types: str or list
            The types of exposure categories to get. Can be 'all_buildings'.

        Returns
        -------
        The list of corresponding exposure categories.
        """
        columns = self.exposure_categories

        if isinstance(types, str):
            types = [types]

        for cat_type in types:
            if cat_type.lower() == 'all_buildings':
                columns = [i for i in columns if i in ['all_buildings']]
                continue
            else:
                raise ValueError(f"Unknown exposure type: {cat_type}")

        return columns

    def _extract_exposure_data(self, directory):
        """
        Extract all contract data.
        """
        exposure_file = glob.glob(directory + '/gvz_exposure*.nc')
        assert len(exposure_file) == 1
        data = self._parse_exposure_files(exposure_file)

        return [data]

    def _parse_exposure_files(self, files):
        """
        Parse the provided exposure files.
        """
        file = files[0]
        with nc4.Dataset(file) as dataset:
            self.domain.check_resolution(dataset, file)
            self._check_extent(dataset, file)
            data = dataset.variables['number_of_buildings'][:]
            self._check_shape(data[0, :, :], file)

            # Convert dates to datetime
            time_data = dataset.variables['time']
            time_units = time_data.units
            time_calendar = time_data.calendar
            time = nc4.num2date(time_data[:], units=time_units, calendar=time_calendar)

            # Extract years and check with the period of interest
            years = [t.year for t in time]

            if self.year_start < min(years):
                raise RuntimeError(f"The starting year {self.year_start} is before the "
                                   f"first year of the data {min(years)}.")
            if self.year_end > max(years):
                raise RuntimeError(f"The ending year {self.year_end} is after the "
                                   f"last year of the data {max(years)}.")

            # Extract the data for the period of interest
            i_start = years.index(self.year_start)
            i_end = years.index(self.year_end)
            data = data[i_start:i_end + 1, :, :]

            return data

    def _extract_claim_data(self, directory):
        """
        Extracts all claims data.
        """
        files = glob.glob(directory + '/gvz_flood_claims*.nc')
        assert len(files) == 1
        self._parse_claim_files(files)

    def _parse_claim_files(self, files):
        """
        Parse the claim files.
        """
        file = files[0]
        with nc4.Dataset(file) as dataset:
            self.domain.check_resolution(dataset, file)
            self._check_extent(dataset, file)
            data = dataset.variables['claim_count'][:]
            self._check_shape(data[0, 0, :, :], file)

            # Convert dates to datetime
            time_data = dataset.variables['time']
            time_units = time_data.units
            time_calendar = time_data.calendar
            time = nc4.num2date(time_data[:], units=time_units, calendar=time_calendar)

            # Extract years and check with the period of interest
            years = [t.year for t in time]

            if self.year_start < min(years):
                raise RuntimeError(f"The starting year {self.year_start} is before the "
                                   f"first year of the data {min(years)}.")
            if self.year_end > max(years):
                raise RuntimeError(f"The ending year {self.year_end} is after the "
                                   f"last year of the data {max(years)}.")

            # Extract the data for the period of interest
            i_start = years.index(self.year_start)
            # Get index of the last occurrence of the year
            i_end = len(years) - 1 - years[::-1].index(self.year_end)

            dates = time[i_start:i_end + 1]
            data = data[:, i_start:i_end + 1, :, :]
            types = dataset.variables['type'][:]

            for i_cat, cat in enumerate(types):
                df_claims = pd.DataFrame(columns=['date_claim', 'mask_index', cat])
                df_claims = df_claims.astype('int32')
                df_claims['date_claim'] = pd.to_datetime(df_claims['date_claim'])
                for i_date, date in enumerate(dates):
                    date = datetime(date.year, date.month, date.day).date()
                    indices, values = self._extract_non_null_claims(
                        data[i_cat, i_date, :, :])
                    df_case = pd.DataFrame(columns=['date_claim', 'mask_index', cat])
                    df_case['date_claim'] = [date] * len(indices)
                    df_case['mask_index'] = indices
                    df_case[cat] = values
                    df_case = df_case.dropna(axis=1, how='all')
                    if not df_case.empty:
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
