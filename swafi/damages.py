"""
Class to handle all contracts and claims.
"""

import glob
import pickle
import ntpath
from datetime import datetime, timedelta, date
from pathlib import Path

import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm

from swafi.config import Config
from swafi.domain import Domain

config = Config()


class Damages:
    def __init__(self, cid_file=None, year_start=None, year_end=None, use_dump=True,
                 dataset='mobi_2023', dir_contracts=None, dir_claims=None,
                 pickle_file=None, pickle_dir=None):
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
        dataset: str
            The dataset ID (default 'mobi_2023')
        dir_contracts: str
            The path to the directory containing the contract files.
        dir_claims: str
            The path to the directory containing the claim files.
        pickle_file: str
            The path to a pickle file to load.
        pickle_dir: str
            The path to the working directory for pickle files
        """
        self.use_dump = use_dump
        self.pickles_dir = pickle_dir
        if pickle_dir is None:
            self.pickles_dir = config.get('PICKLES_DIR')

        self.domain = Domain(cid_file)
        self.cids_list = None
        self.mask = dict(extent=None, shape=None, mask=np.array([]), xs=np.array([]),
                         ys=np.array([]))

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

            self.tags_contracts = [
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

        self.contracts = pd.DataFrame(
            columns=['year', 'mask_index', 'selection'] + self.categories)
        self.claims = pd.DataFrame(
            columns=['date_claim', 'mask_index', 'selection'])

        self.contracts = self.contracts.astype('int32')
        self.claims = self.claims.astype('int32')
        self.claims['date_claim'] = pd.to_datetime(self.claims['date_claim'])

        self._load_from_dump()

        if dir_contracts is not None:
            self.load_contracts(dir_contracts)

        if dir_claims is not None:
            self.load_claims(dir_claims)

        if pickle_file is not None:
            self.load_from_pickle(pickle_file)

    def load_from_pickle(self, filename='damages.pickle'):
        """
        Load the damage pre-processed data from a pickle file.

        Parameters
        ----------
        filename: str
            The name the pickle file in the project temporary directory (PICKLES_DIR)
        """
        self._load_from_dump(filename=filename)

    def load_contracts(self, directory=None):
        """
        Load the contract data from geotiff files.

        Parameters
        ----------
        directory: str
            The path to the directory containing the files.
        """
        if self.use_dump and self.mask['mask'].size > 0:
            print("Contracts reloaded from pickle file.")
            return

        if not directory:
            directory = config.get('DIR_CONTRACTS')

        contract_data = self._extract_contract_data(directory)
        self._create_mask(contract_data)
        self._create_cids_list()

        for idx, contracts in enumerate(contract_data):
            contract_data_cat = self._extract_data_with_mask(contracts)
            if idx == 0:
                self._initialize_contracts_dataframe(contract_data_cat)
            self._set_to_contracts_dataframe(contract_data_cat, self.categories[idx])

        self._set_contracts_cids()

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
        self._set_claims_cids()

        self._dump_object()

    def set_target_variable_value(self, mode='occurrence'):
        """
        Set the target variable value.

        Parameters
        ----------
        mode : str
            The mode to set the target variable. Can be 'occurrence' or 'ratio'.
        """
        self.claims['target'] = 0
        if mode == 'occurrence':
            self.claims.loc[self.claims.selection > 0, 'target'] = 1
        elif mode == 'ratio':
            self.claims.target = self.claims.selection
            raise NotImplementedError

    def select_all_categories(self):
        """
        Select all the damage categories.
        """
        columns = self.categories
        self.contracts['selection'] = self.contracts[columns].sum(axis=1)
        self.claims['selection'] = self.claims[columns].sum(axis=1)

    def categories_are_for_type(self, types):
        """
        Check if the categories are for a given type.

        Parameters
        ----------
        types: str or list
            The types of categories to check. Can be 'external', 'internal', 'sme',
            'private', 'content', 'structure'.

        Returns
        -------
        True if the categories are for the given type, False otherwise.
        """
        categories = self.get_categories_from_type(types)

        return categories == self.selected_categories

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

    def select_categories_type(self, types):
        """
        Select the damage categories corresponding to a certain type.

        Parameters
        ----------
        types: list|str
            The types of the damage categories to select. The type are exclusive.
            For example : ['external', 'structure'].
            Options are:
            - 'external' vs 'internal' (the building)
            - 'SME' vs 'private'
            - 'content' vs 'structure' (of the building)

        Returns
        -------
        The list of selected categories.
        """
        columns = self.get_categories_from_type(types)
        self.contracts['selection'] = self.contracts[columns].sum(axis=1)
        self._apply_categories_selection(columns)

        return columns

    def select_categories(self, categories):
        """
        Select the given damage categories.

        Parameters
        ----------
        categories: list
            A list of the categories to select. For example, for the 'mobi_2023' dataset
            the possible categories are: 'sme_ext_cont', 'sme_ext_struc',
            'sme_int_cont', 'sme_int_struc', 'priv_ext_cont', 'priv_ext_struc',
            'priv_int_cont', 'priv_int_struc'
        """
        self.contracts['selection'] = self.contracts[categories].sum(axis=1)
        self._apply_categories_selection(categories)

    def link_with_events(self, events, criteria=None, window_days=None,
                         filename='damages_matched.pickle'):
        """
        Link the damages with the events.

        Parameters
        ----------
        events: Events instance
            An object containing the events properties.
        criteria: list (optional)
            A list of the criteria to consider for the matching.
            Default to ['i_mean', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt']
            where:
            - i_mean: mean intensity of the event
            - i_max: max intensity of the event
            - p_sum: sum of the event precipitation
            - r_ts_win: ratio of the event time steps within the temporal window on the
              total window duration
            - r_ts_evt: ratio of the event time steps within the temporal window on the
              total event duration
            - prior: put more weights on events occurring prior to the claim
        window_days: list (optional)
            A list of the temporal window (days) on which to search for events to match.
            Default to [5, 3, 1]
        filename: str
            File name to save the results (pickle format)
        """
        if window_days is None:
            window_days = [5, 3, 1]
        if criteria is None:
            criteria = ['i_mean', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt']

        self._add_event_matching_fields(events, window_days, criteria)
        stats = dict(none=0, single=0, two=0, three=0, multiple=0,
                     conflicts=0, unresolved=0)

        for i_claim in tqdm(range(len(self.claims)), desc=f"Matching claims / events"):
            claim = self.claims.iloc[i_claim]

            # Get potential events
            pot_events = self._get_potential_events(claim, events, window_days)
            self._record_stat_candidates(stats, pot_events)

            if pot_events is None:
                continue

            # Assign points for all windows and criteria
            self._compute_match_score(claim, criteria, pot_events, window_days)

            # Getting the best event matches
            best_matches = self._get_best_candidate(pot_events, window_days, stats)
            self._record_best_event(best_matches, i_claim)

        self._print_matches_stats(stats)
        self._remove_claims_with_no_event()
        self._dump_object(filename)

    def merge_with_events(self, events):
        """
        Merge the claims with the pre-assigned event data using the fields 'cid' and
        'eid'. The prior use of the match_with_events() function is mandatory to assign
        the event IDs (eid).

        Parameters
        ----------
        events: Events instance
            An object containing the events properties.
        """
        self.claims = pd.merge(self.claims, events.events,
                               how="left", on=['cid', 'eid'])

    def compute_days_to_event_start(self, field_name='dt_start'):
        """
        Compute the number of days between the claims and the events start.
        The prior use of the match_with_events() and the merge_with_events()
        functions is mandatory.

        Parameters
        ----------
        field_name: str
            The name of the field to add to the dataframe.
        """
        claims = self.claims
        self.claims[field_name] = claims.e_start.dt.date - claims.date_claim
        self.claims[field_name] = claims[field_name].apply(lambda x: x.days)

    def compute_days_to_event_center(self, field_name='dt_center'):
        """
        Compute the number of days between the claims and the events center (average
        of the start and end of the event). The prior use of the match_with_events()
        and the merge_with_events() functions is mandatory.

        Parameters
        ----------
        field_name: str
            The name of the field to add to the dataframe.
        """
        claims = self.claims
        self.claims[field_name] = ((claims.e_start + (
                claims.e_end - claims.e_start) / 2).dt.date - claims.date_claim)
        self.claims[field_name] = claims[field_name].apply(lambda x: x.days)

    def _apply_categories_selection(self, categories):
        self.contracts = self.contracts[self.contracts.selection != 0]
        self.contracts.reset_index(inplace=True, drop=True)
        self.claims['selection'] = self.claims[categories].sum(axis=1)
        self.claims = self.claims[self.claims.selection != 0]
        self.claims.reset_index(inplace=True, drop=True)
        self.selected_categories = categories

    def _add_event_matching_fields(self, events, window_days, criteria):
        self.claims.reset_index(inplace=True, drop=True)
        self.claims['eid'] = 0
        self.claims['e_search_window'] = 0
        self.claims['e_match_score'] = 0

        events.events['min_window'] = 0
        events.events['match_score'] = 0
        events.events['prior'] = 0
        events.events['overlap_hrs'] = 0
        events.events['r_ts_win'] = 0  # former 'tx', (#ts overlap)/(#ts window)
        events.events['r_ts_evt'] = 0  # former 'et', (#ts overlap)/(evt duration)

        for window in window_days:
            for criterion in criteria:
                if criterion == "prior":
                    continue
                field_name = f'{criterion}_{window}'
                events.events[field_name] = 0

    def _remove_claims_with_no_event(self):
        self.claims = self.claims[self.claims.eid != 0]
        self.claims.reset_index(inplace=True, drop=True)

    @staticmethod
    def _get_best_candidate(pot_events, window_days, stats):
        best_matches = pot_events.loc[pot_events['match_score'] ==
                                      pot_events['match_score'].max()].copy()

        if len(best_matches) > 1:
            stats['conflicts'] += 1
            for window in reversed(window_days):
                best_matches['sub_score'] = 0
                if f'i_mean_{window}' in best_matches:
                    best_matches['sub_score'] += best_matches[f'i_mean_{window}']
                if f'i_max_{window}' in best_matches:
                    best_matches['sub_score'] += best_matches[f'i_max_{window}']
                if f'p_sum_{window}' in best_matches:
                    best_matches['sub_score'] += best_matches[f'p_sum_{window}']

                best_matches = best_matches.loc[best_matches['sub_score'] ==
                                                best_matches['sub_score'].max()]
                if len(best_matches) == 1:
                    break

        if len(best_matches) > 1:
            stats['unresolved'] += 1
            best_matches = best_matches.head(1)

        return best_matches

    def _record_best_event(self, best_matches, i_claim):
        self.claims.at[i_claim, 'eid'] = best_matches.iloc[0].eid
        self.claims.at[i_claim, 'e_search_window'] = best_matches.iloc[0].min_window
        self.claims.at[i_claim, 'e_match_score'] = best_matches.iloc[0].match_score

    def _compute_match_score(self, claim, criteria, pot_events, window_days):
        if 'prior' in criteria:
            self._compute_prior_to_claim(claim['date_claim'], pot_events)
            pot_events.loc[pot_events['prior'] == 1, 'match_score'] += 1

        for window in window_days:
            if 'r_ts_win' in criteria or 'r_ts_evt' in criteria:
                self._compute_temporal_overlap(claim['date_claim'], pot_events, window)
                pot_events['r_ts_win'] = pot_events['overlap_hrs'] / (window * 24)
                pot_events['r_ts_evt'] = pot_events['overlap_hrs'] / pot_events['e_tot']
            for criterion in criteria:
                if criterion == 'prior':
                    continue
                within_window = pot_events['min_window'] <= window
                val_max = pot_events.loc[within_window, criterion].max()
                with_max_val = pot_events[criterion] == val_max
                if with_max_val.empty:
                    continue
                field_name = f'{criterion}_{window}'
                pot_events.loc[within_window & with_max_val, field_name] = 1
                pot_events.loc[within_window & with_max_val, 'match_score'] += 1

    @staticmethod
    def _record_stat_candidates(stats, pot_events):
        if pot_events is None:
            stats['none'] += 1
        elif len(pot_events) == 1:
            stats['single'] += 1
        elif len(pot_events) == 2:
            stats['two'] += 1
        elif len(pot_events) == 3:
            stats['three'] += 1
        else:
            stats['multiple'] += 1

    @staticmethod
    def _print_matches_stats(stats):
        print(f"Stats of the events / damage matches:")
        print(f"- {stats['none']} claims could not be matched")
        print(f"- {stats['single']} claims had 1 candidate event")
        print(f"- {stats['two']} claims had 2 candidate events")
        print(f"- {stats['three']} claims had 3 candidate events")
        print(f"- {stats['multiple']} claims had more candidate event")
        print(f"- {stats['conflicts']} claims had conflicts")
        print(f"- {stats['unresolved']} matching were unresolved (first event taken)")

    @staticmethod
    def _compute_temporal_overlap(date_claim, pot_events, window):
        delta_days = (window - 1) / 2
        date_window_start = datetime.combine(
            date_claim - timedelta(days=delta_days),
            datetime.min.time())
        date_window_end = datetime.combine(
            date_claim + timedelta(days=delta_days),
            datetime.max.time())
        for i, event in pot_events.iterrows():
            e_start_corr = event['e_start'] - timedelta(hours=1)  # 1 hr is missing
            overlap_window_start = max(date_window_start, e_start_corr)
            overlap_window_end = min(date_window_end, event['e_end'])
            overlap = overlap_window_end - overlap_window_start
            overlap_hrs = max(0.0, overlap.total_seconds() / 3600)
            pot_events.at[i, 'overlap_hrs'] = overlap_hrs

    @staticmethod
    def _compute_prior_to_claim(date_claim, pot_events):
        date_claim_end_day = datetime.combine(
            date_claim, datetime.max.time())
        for i, event in pot_events.iterrows():
            if event['e_start'] < date_claim_end_day:
                pot_events.at[i, 'prior'] = 1

    @staticmethod
    def _get_potential_events(claim, events, window_days):
        """
        Get all potential events based on the CID and the date.
        """
        cid = claim['cid']
        date_claim = claim['date_claim']

        # Define the starting and ending dates of the longest temporal window
        window_days.sort(reverse=True)
        date_window_end, date_window_start = Damages._get_window_dates(
            date_claim, max(window_days))

        # Select all events in the longest temporal window
        potential_events = events.events[
            (events.events['cid'] == cid) &
            (events.events['e_start'] < date_window_end) &
            (events.events['e_end'] > date_window_start)]

        if len(potential_events) == 0:
            return None

        potential_events = potential_events.copy()
        potential_events['min_window'] = window_days[0]

        # Assess all other temporal windows and keep the smallest value
        for window in window_days[1:]:
            date_window_end, date_window_start = Damages._get_window_dates(
                date_claim, window)
            potential_events.loc[
                (potential_events['e_start'] < date_window_end) &
                (potential_events['e_end'] > date_window_start),
                'min_window'] = window

        return potential_events

    @staticmethod
    def _get_window_dates(date_claim, window):
        if (window % 2) == 0:  # Even number: use day and day-1 as center
            delta_days = (window - 2) / 2
            date_window_start = datetime.combine(
                date_claim - timedelta(days=delta_days + 1),
                datetime.min.time())
        else:
            delta_days = (window - 1) / 2
            date_window_start = datetime.combine(
                date_claim - timedelta(days=delta_days),
                datetime.min.time())
        date_window_end = datetime.combine(
            date_claim + timedelta(days=delta_days),
            datetime.max.time())

        return date_window_end, date_window_start

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

    def _store_in_claims_dataframe(self, df_claims):
        """
        Stores the claims for a given category in the dataframe.
        """
        self.claims = pd.merge(self.claims, df_claims, how='outer',
                               on=['date_claim', 'mask_index'], validate='one_to_one')

    def _extract_non_null_claims(self, data):
        """
        Extracts the cells with at least 1 claim.
        """
        # Extract the pixels where the catalog is not null
        extracted = np.extract(self.mask['mask'], data[0, :, :])
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

    def _check_extent(self, dataset, file):
        """
        Check extent consistency with other files.
        """
        if self.mask['extent'] is None:
            self.mask['extent'] = dataset.bounds

            # Extract the axes
            data = dataset.read()
            data = data.squeeze(axis=0)
            height = data.shape[0]
            width = data.shape[1]
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            xs, ys = rasterio.transform.xy(dataset.transform, rows, cols)
            self.mask['xs'] = np.array(xs)
            self.mask['ys'] = np.array(ys)

        elif self.mask['extent'] != dataset.bounds:
            raise RuntimeError(f"The extent of {file} differs from other files.")

    def _check_shape(self, data, file):
        """
        Check shape consistency with other files.
        """
        if self.mask['shape'] is None:
            self.mask['shape'] = data.shape
        elif self.mask['shape'] != data.shape:
            raise RuntimeError(f"The shape of {file} differs from other files.")

    def _create_mask(self, contract_data):
        """
        Creates a mask with True for all pixels containing at least 1 annual contract.
        """
        self.mask['mask'] = np.zeros(self.mask['shape'][1:], dtype=bool)
        for arr in contract_data:
            max_value = arr.max(axis=0)
            self.mask['mask'][max_value > 0] = True

    def _create_cids_list(self):
        """
        Creates the CIDs list for cells where we have damages
        """
        xs_mask = np.extract(self.mask['mask'], self.mask['xs'])
        ys_mask = np.extract(self.mask['mask'], self.mask['ys'])

        cids = np.zeros(len(xs_mask))
        xs_cid = self.domain.cids['xs'][0, :]
        ys_cid = self.domain.cids['ys'][:, 0]

        for i, (x, y) in enumerate(zip(xs_mask, ys_mask)):
            cids[i] = self.domain.cids['ids_map'][ys_cid == y, xs_cid == x]

        self.cids_list = cids

    def _extract_data_with_mask(self, data):
        """
        Extracts data according to the mask and returns a 1-D array.
        """
        if self.mask['mask'].size == 0:
            raise RuntimeError("The mask for extraction was not defined.")
        extracted = np.zeros((data.shape[0], np.sum(self.mask['mask'])), dtype=np.int16)
        for i in range(data.shape[0]):
            extracted[i, :] = np.extract(self.mask['mask'], data[i, :, :])
        return extracted

    def _load_from_dump(self, filename='damages.pickle'):
        """
        Loads the object content from a pickle file.
        """
        if not self.use_dump:
            return
        file_path = Path(self.pickles_dir + '/' + filename)
        if file_path.is_file():
            with open(file_path, 'rb') as f:
                values = pickle.load(f)
                self.mask = values.mask
                self.contracts = values.contracts
                self.claims = values.claims
                self.cids_list = values.cids_list
                if hasattr(values, 'selected_categories'):
                    self.selected_categories = values.selected_categories

    def _dump_object(self, filename='damages.pickle'):
        """
        Saves the object content to a pickle file.
        """
        if not self.use_dump:
            return
        file_path = Path(self.pickles_dir + '/' + filename)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def _initialize_contracts_dataframe(self, contract_data_cat):
        """
        Initializes the contracts dataframe by filling the year and the mask_index columns.
        The mask_index column refers to the 1-D array after extraction by the mask.
        """
        n_years = self.year_end - self.year_start + 1
        n_annual_rows = contract_data_cat.shape[1]
        years = np.repeat(np.arange(self.year_start, self.year_end + 1), n_annual_rows)
        self.contracts['year'] = years
        indices = np.tile(np.arange(n_annual_rows), n_years)
        self.contracts['mask_index'] = indices

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
        columns = ['date_claim', 'mask_index', 'selection'] + self.categories
        self.claims = self.claims.reindex(columns=columns)
        self.claims.fillna(0, inplace=True)
        self.claims.sort_values(by=['date_claim', 'mask_index'], inplace=True)
        self.claims.reset_index(inplace=True, drop=True)
        for category in self.categories:
            self.claims[category] = self.claims[category].astype('int32')

    def _set_claims_cids(self):
        xs_mask_extracted = np.extract(self.mask['mask'], self.mask['xs'])
        ys_mask_extracted = np.extract(self.mask['mask'], self.mask['ys'])
        cids = self.cids_list[self.claims['mask_index']].astype(np.int32)
        x = xs_mask_extracted[self.claims['mask_index']].astype(np.int32)
        y = ys_mask_extracted[self.claims['mask_index']].astype(np.int32)
        self.claims.insert(2, 'cid', cids)
        self.claims.insert(3, 'x', x)
        self.claims.insert(4, 'y', y)

    def _set_contracts_cids(self):
        xs_mask_extracted = np.extract(self.mask['mask'], self.mask['xs'])
        ys_mask_extracted = np.extract(self.mask['mask'], self.mask['ys'])
        cids = self.cids_list[self.contracts['mask_index']].astype(np.int32)
        x = xs_mask_extracted[self.contracts['mask_index']].astype(np.int32)
        y = ys_mask_extracted[self.contracts['mask_index']].astype(np.int32)
        self.contracts.insert(2, 'cid', cids)
        self.contracts.insert(3, 'x', x)
        self.contracts.insert(4, 'y', y)
