"""
Class to handle all exposure and claims.
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import rasterio
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

try:
    import netCDF4 as nc4
except ImportError:
    nc4 = None

from .config import Config
from .domain import Domain

config = Config()

# Temporal window for claim-event matching (hours relative to the claim day at 0 h)
SIMPLE_EVENT_HOURS_BEFORE = 8
SIMPLE_EVENT_HOURS_AFTER = 26

logger = logging.getLogger(__name__)


class Damages:
    def __init__(self, cid_file=None, year_start=None, year_end=None, use_dump=True,
                 pickle_dir=None):
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
        pickle_dir: str
            The path to the working directory for pickle files
        """
        self.name = None  # Name of the dataset to be defined in the child class
        self.use_dump = use_dump
        self.pickles_dir = pickle_dir
        if pickle_dir is None:
            self.pickles_dir = config.get('PICKLES_DIR')

        self.domain = Domain(cid_file)
        self.cids_list = None
        self.mask = dict(extent=(None, None, None, None), shape=None, mask=np.array([]),
                         xs=np.array([]), ys=np.array([]))

        self.year_start = year_start
        if not self.year_start:
            self.year_start = config.get('YEAR_START', 2013)
        self.year_end = year_end
        if not self.year_end:
            self.year_end = config.get('YEAR_END', 2022)

        self.claim_categories = []
        self.exposure_categories = []
        self.selected_claim_categories = []
        self.selected_exposure_categories = []

    def load_from_pickle(self, filename):
        """
        Load the damage pre-processed data from a pickle file.

        Parameters
        ----------
        filename: str
            The name the pickle file in the project temporary directory (PICKLES_DIR)
        """
        self._load_from_dump(filename=filename)

    def load_exposure(self, directory=None):
        """
        Load the exposure data from geotiff files.

        Parameters
        ----------
        directory: str
            The path to the directory containing the files.
        """
        if self.use_dump and self.mask['mask'].size > 0:
            logger.info("Exposure files reloaded from pickle file.")
            return

        if not directory:
            directory = config.get('DIR_EXPOSURE')

        exposure_data = self._extract_exposure_data(directory)
        self._create_mask(exposure_data)
        self._create_cids_list()

        for idx, contracts in enumerate(exposure_data):
            exposure_data_cat = self._extract_data_with_mask(contracts)
            if idx == 0:
                self._initialize_exposure_dataframe(exposure_data_cat)
            self._set_to_exposure_dataframe(
                exposure_data_cat, self.exposure_categories[idx])

        self._set_exposure_cids()

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
            logger.info("Claims reloaded from pickle file.")
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
            The mode to set the target variable. Can be 'occurrence' or 'damage_ratio'.
        """
        self.claims['target'] = 0
        if mode == 'occurrence':
            self.claims.loc[self.claims.selection > 0, 'target'] = 1
        elif mode == 'damage_ratio':
            self._compute_claim_exposure_ratio()

    def select_all_categories(self):
        """
        Select all the damage categories.
        """
        self.exposure['selection'] = self.exposure[self.claim_categories].sum(axis=1)
        self.claims['selection'] = self.claims[self.exposure_categories].sum(axis=1)

    def exposure_categories_are_for_type(self, types):
        """
        Check if the exposure categories are for a given type.

        Parameters
        ----------
        types: str or list
            The types of exposure categories to check. Depends on the dataset.

        Returns
        -------
        True if the exposure categories are for the given type, False otherwise.
        """
        categories = self.get_exposure_categories_from_type(types)

        return categories == self.selected_exposure_categories

    def claim_categories_are_for_type(self, types):
        """
        Check if the claim categories are for a given type.

        Parameters
        ----------
        types: str or list
            The types of claim categories to check. Depends on the dataset.

        Returns
        -------
        True if the claim categories are for the given type, False otherwise.
        """
        categories = self.get_claim_categories_from_type(types)

        return categories == self.selected_claim_categories

    def get_exposure_categories_from_type(self, types):
        raise NotImplementedError("This method should be implemented "
                                  "in the child class.")

    def get_claim_categories_from_type(self, types):
        raise NotImplementedError("This method should be implemented "
                                  "in the child class.")

    def select_categories_type(self, exposure_types, claim_types):
        """
        Select the damage categories corresponding to a certain type.

        Parameters
        ----------
        exposure_types: list|str
            The types of the claim categories to select. The type are exclusive.
            For example : ['external', 'structure'].
            Options are dependent on the dataset.
        claim_types: list|str
            The types of the exposure categories to select. The type are exclusive.
            Options are dependent on the dataset.

        Returns
        -------
        The claims that have been removed from the dataset.
        """
        columns_exposure = self.get_exposure_categories_from_type(exposure_types)
        self.selected_exposure_categories = columns_exposure
        columns_claims = self.get_claim_categories_from_type(claim_types)
        self.exposure['selection'] = self.exposure[columns_exposure].sum(axis=1)

        return self._apply_claim_categories_selection(columns_claims)

    def select_claim_categories(self, categories):
        """
        Select the given claim categories.

        Parameters
        ----------
        categories: list
            A list of the categories to select. For example, for the Mobiliar dataset
            the possible categories are: 'sme_ext_cont', 'sme_ext_struc',
            'sme_int_cont', 'sme_int_struc', 'priv_ext_cont', 'priv_ext_struc',
            'priv_int_cont', 'priv_int_struc'

        Returns
        -------
        The claims that have been removed from the dataset.
        """
        return self._apply_claim_categories_selection(categories)

    def select_exposure_categories(self, categories):
        """
        Select the given exposure categories.

        Parameters
        ----------
        categories: list
            A list of the categories to select.
        """
        self.exposure['selection'] = self.exposure[categories].sum(axis=1)

    def link_with_events(self, events, method='simple', criteria=None, window_days=None,
                         filename=None):
        """
        Link the damages with the events.

        Parameters
        ----------
        events: Events instance
            An object containing the events properties.
        method: str
            The method to use for the events extraction. Can be 'simple' or 'classic'.
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

        Returns
        -------
        The list of events to remove from the events dataframe
        """
        events_to_remove = []
        if filename is None:
            filename = f'damages_{self.name}_matched.pickle'

        if method == 'classic':
            if window_days is None:
                window_days = [5, 3, 1]
            if criteria is None:
                criteria = ['i_mean', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt']
            # Sorted copy: do not mutate the caller's list
            window_days = sorted(window_days, reverse=True)

            self._add_event_matching_fields(events, window_days, criteria)
            stats = dict(none=0, single=0, two=0, three=0, multiple=0,
                         conflicts=0, unresolved=0)

            # Group the events per cell (sorted by start date): the candidate
            # lookup per claim is then a binary search instead of a scan of
            # the whole events dataframe.
            events_by_cid = self._group_events_by_cid(events.events, 'e_start')
            max_duration = (events.events['e_end'] - events.events['e_start']).max()

            for i_claim in tqdm(range(len(self.claims)), desc=f"Matching claim/events"):
                claim = self.claims.iloc[i_claim]

                # Get potential events
                pot_events = self._get_potential_classic_events(
                    claim, events_by_cid, window_days, max_duration)
                self._record_stat_candidates(stats, pot_events)

                if pot_events is None:
                    continue

                # Assign points for all windows and criteria
                self._compute_match_score(claim, criteria, pot_events, window_days)

                # Getting the best event matches
                best_matches = self._get_best_candidate(pot_events, window_days, stats)
                self._record_best_event(best_matches, i_claim)

                # Remove the events that have been matched
                if len(pot_events) > 1:
                    ev_to_remove = pot_events.eid.tolist()
                    best_eid = best_matches.eid.tolist()[0]
                    ev_to_remove.remove(best_eid)
                    events_to_remove.extend(ev_to_remove)

            # Check again that the events to remove were not selected in the claims
            claim_eids = set(self.claims['eid'].tolist())
            events_to_remove = [ev for ev in events_to_remove if
                                ev not in claim_eids]
            logger.info("Events to remove due to claim/event link: %s", len(events_to_remove))

            self._print_matches_stats(stats)

        elif method == 'simple':
            stats = dict(none=0, single=0, two=0, three=0, multiple=0)

            # The 'eid' column must exist (as int) before the loop: otherwise
            # the .at setter creates a float column where unmatched claims end
            # up as NaN, which the eid != 0 filter below would keep.
            self.claims.reset_index(inplace=True, drop=True)
            self.claims['eid'] = 0

            # Group the events per cell (sorted by i_max_date) for fast lookup
            events_by_cid = self._group_events_by_cid(events.events, 'i_max_date')

            for i_claim in tqdm(range(len(self.claims)), desc=f"Matching claim/events"):
                claim = self.claims.iloc[i_claim]

                # Get potential events
                pot_events = self._get_potential_simple_events(claim, events_by_cid)
                self._record_stat_candidates(stats, pot_events)

                if pot_events is None:
                    continue

                best_match = self._get_best_candidate_simple(pot_events, claim)
                self.claims.at[i_claim, 'eid'] = best_match.eid

                # Remove the events that have been matched
                if len(pot_events) > 1:
                    ev_to_remove = pot_events.eid.tolist()
                    ev_to_remove.remove(best_match.eid)
                    events_to_remove.extend(ev_to_remove)

            # Check again that the events to remove were not selected in the claims
            claim_eids = set(self.claims['eid'].tolist())
            events_to_remove = [ev for ev in events_to_remove if
                                ev not in claim_eids]
            logger.info("Events to remove due to claim/event link: %s", len(events_to_remove))

            self._print_matches_stats(stats)

        else:
            raise ValueError(f"Unknown method: {method}")

        self._remove_claims_with_no_event()
        self._dump_object(filename)

        return events_to_remove

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
        self.claims[field_name] = (pd.to_datetime(claims.e_start) -
                                   pd.to_datetime(claims.date_claim)).dt.days

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
        midpoint_date = claims.e_start + (claims.e_end - claims.e_start) / 2
        self.claims[field_name] = (midpoint_date - claims.date_claim).dt.days

    def to_xarray(self, save_to_nc=True, removed_claims=None):
        """
        Convert the exposure and claims dataframes to xarray datasets.

        Parameters
        ----------
        save_to_nc: bool
            Whether to save the datasets to netCDF4 files. Default is True.
        removed_claims: DataFrame
            The claims that have been removed from the dataset when selecting
            the claim categories.

        Returns
        -------
        xr.Dataset
            The xarray dataset containing the exposure and claims data.
        """
        # Pickle file path
        pickle_path = Path(self.pickles_dir) / f'damages_{self.name}_{self.year_start}_{self.year_end}_xr.pickle'
        if self.use_dump and pickle_path.exists():
            with open(pickle_path, 'rb') as f:
                claims_ds = pickle.load(f)
            logger.info("Claims datasets reloaded from pickle file: %s", pickle_path)
            return claims_ds

        # Create daily time axis
        time = pd.date_range(
            start=f"{self.year_start}-01-01",
            end=f"{self.year_end}-12-31",
            freq="D"
        )

        # Get spatial axes
        xs = self.domain.get_x_axis()
        ys = self.domain.get_y_axis()

        # Numpy buffers (wrapped into DataArrays at the end)
        claims_np = np.full((len(time), len(ys), len(xs)), np.nan, dtype=np.float32)
        exposure_np = np.full_like(claims_np, np.nan)

        # Set values to 0 where there is exposure (vectorized per year)
        exposure = self.exposure[self.exposure['selection'] != 0]
        if not exposure.empty:
            exp_x = self._nearest_axis_indices(xs, exposure['x'].to_numpy())
            exp_y = self._nearest_axis_indices(ys, exposure['y'].to_numpy())
            exp_values = exposure['selection'].to_numpy()
            for year, rows in exposure.groupby('year').indices.items():
                t_sel = np.where(time.year == year)[0]
                if t_sel.size == 0:
                    continue
                exposure_np[t_sel[:, None], exp_y[rows][None, :],
                            exp_x[rows][None, :]] = exp_values[rows][None, :]
                claims_np[t_sel[:, None], exp_y[rows][None, :],
                          exp_x[rows][None, :]] = 0.0

        def _scatter_claims(target, claims_df, values):
            t_idx = time.searchsorted(pd.to_datetime(claims_df['date_claim']))
            x_idx = self._nearest_axis_indices(xs, claims_df['x'].to_numpy())
            y_idx = self._nearest_axis_indices(ys, claims_df['y'].to_numpy())
            in_range = t_idx < len(time)
            target[t_idx[in_range], y_idx[in_range], x_idx[in_range]] = \
                values[in_range]

        # Place each claim's selection value
        if not self.claims.empty:
            _scatter_claims(claims_np, self.claims,
                            self.claims['selection'].to_numpy())

        coords = {"time": time, "y": ys, "x": xs}
        dims = ["time", "y", "x"]
        claims_da = xr.DataArray(claims_np, coords=coords, dims=dims, name="claims")
        exposure_da = xr.DataArray(exposure_np, coords=coords, dims=dims,
                                   name="exposure")

        # Combine into a single dataset
        xr_ds = xr.Dataset({"claims": claims_da, "exposure": exposure_da})
        if removed_claims is not None and not removed_claims.empty:
            removed_np = np.full_like(claims_np, np.nan)
            _scatter_claims(removed_np, removed_claims,
                            np.ones(len(removed_claims), dtype=np.float32))
            xr_ds["removed_claims"] = xr.DataArray(
                removed_np, coords=coords, dims=dims, name="removed_claims")
        xr_ds.attrs['year_start'] = self.year_start
        xr_ds.attrs['year_end'] = self.year_end
        xr_ds.attrs['exposure_categories'] = self.selected_exposure_categories
        xr_ds.attrs['claim_categories'] = self.selected_claim_categories
        xr_ds.attrs['dataset_name'] = self.name
        xr_ds.attrs['creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        xr_ds.attrs['source'] = 'Generated by SWAFI damages.to_xarray()'

        # Save to pickle
        if self.use_dump:
            with open(pickle_path, 'wb') as f:
                pickle.dump(xr_ds, f)
            logger.info("Claims datasets saved to pickle file: %s", pickle_path)

        # Save to netCDF4
        if save_to_nc and nc4 is not None:
            netcdf_path = Path(self.pickles_dir) / f'damages_{self.name}_{self.year_start}_{self.year_end}.nc'
            xr_ds.to_netcdf(netcdf_path)
            logger.info("Claims datasets saved to netCDF4 file: %s", netcdf_path)

        return xr_ds

    def from_nc_file(self, filepath):
        """
        Load the exposure and claims data from a netCDF4 file.

        Parameters
        ----------
        filepath: str
            The path to the netCDF4 file.

        Returns
        -------
        xr.Dataset
            The xarray dataset containing the exposure and claims data.
        """
        if nc4 is None:
            raise ImportError("netCDF4 is not installed. Cannot read netCDF files.")

        xr_ds = xr.open_dataset(filepath)
        self.year_start = xr_ds.attrs.get('year_start', self.year_start)
        self.year_end = xr_ds.attrs.get('year_end', self.year_end)
        self.selected_exposure_categories = xr_ds.attrs.get('exposure_categories', [])
        self.selected_claim_categories = xr_ds.attrs.get('claim_categories', [])
        self.name = xr_ds.attrs.get('dataset_name', self.name)

        return xr_ds

    def _create_exposure_claims_df(self):
        self.exposure = pd.DataFrame(
            columns=['year', 'mask_index', 'selection'] + self.exposure_categories)
        self.claims = pd.DataFrame(
            columns=['date_claim', 'mask_index', 'selection'])

        self.exposure = self.exposure.astype('int32')
        self.claims = self.claims.astype('int32')
        self.claims['date_claim'] = pd.to_datetime(self.claims['date_claim'])

    def _apply_claim_categories_selection(self, categories):
        self.exposure = self.exposure[self.exposure.selection != 0]
        self.exposure.reset_index(inplace=True, drop=True)
        self.claims['selection'] = self.claims[categories].sum(axis=1)
        removed_claims = self.claims[self.claims.selection == 0]
        self.claims = self.claims[self.claims.selection != 0]
        self.claims.reset_index(inplace=True, drop=True)
        self.selected_claim_categories = categories

        return removed_claims

    def _compute_claim_exposure_ratio(self):
        # Check for duplicate keys in self.exposure
        duplicate_keys_exposure = self.exposure[
            self.exposure.duplicated(subset=['year', 'cid'], keep=False)]
        if not duplicate_keys_exposure.empty:
            raise ValueError("Duplicate keys in self.exposure")

        # If column nb_contracts does not exist
        if 'nb_contracts' not in self.claims.columns:
            # Extract the fields to compute the ratio
            exposure = self.exposure[['year', 'cid', 'selection']]
            exposure.rename(columns={'selection': 'nb_contracts'}, inplace=True)

            # Extract year from the 'date_claim' column in self.claims
            self.claims['year'] = pd.to_datetime(self.claims['date_claim'])
            self.claims['year'] = self.claims['year'].dt.year

            # Merge the two dataframes
            self.claims = pd.merge(self.claims, exposure, how='left', on=['year', 'cid'])
            self.claims.drop('year', axis=1, inplace=True)

        self.claims.target = self.claims.selection / self.claims.nb_contracts
        self.claims.loc[self.claims['target'] > 1, 'target'] = 1

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

    def _remove_data_outside_period(self):
        # Ensure the 'date_claim' column is of datetime type
        self.claims['date_claim'] = pd.to_datetime(
            self.claims['date_claim'], errors='coerce')
        self.claims = self.claims[
            (self.claims.date_claim.dt.year >= self.year_start) &
            (self.claims.date_claim.dt.year <= self.year_end)]
        self.exposure = self.exposure[
            (self.exposure.year >= self.year_start) &
            (self.exposure.year <= self.year_end)]

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

    def _get_best_candidate_simple(self, pot_events, claim):
        if len(pot_events) == 1:
            return pot_events.iloc[0]

        else:  # 2 or more potential events
            # Select the event(s) with the highest i_max
            best_idx = pot_events.i_max.idxmax()
            best_events = pot_events[pot_events.i_max == pot_events.loc[best_idx].i_max]
            if len(best_events) == 1:
                return best_events.iloc[0]

            # If multiple events have the same i_max, keep the claim date
            best_event = best_events[best_events.e_date == claim.date_claim.floor('D')]
            if best_event.empty:
                # Return the closest event to the claim date
                best_events['date_diff'] = (best_events.e_date - claim.date_claim).abs()
                best_event = best_events.loc[best_events.date_diff.idxmin()]
                return best_event

            return best_event.iloc[0]

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
                pot_events['r_ts_evt'] = pot_events['overlap_hrs'] / pot_events['duration']
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
        logger.info("Stats of the events / damage matches:")
        logger.info("- %s claims could not be matched", stats['none'])
        logger.info("- %s claims had 1 candidate event", stats['single'])
        logger.info("- %s claims had 2 candidate events", stats['two'])
        logger.info("- %s claims had 3 candidate events", stats['three'])
        logger.info("- %s claims had more candidate event", stats['multiple'])
        if 'conflicts' in stats:
            logger.info("- %s claims had conflicts", stats['conflicts'])
            logger.info("- %s matching were unresolved (first event taken)", stats['unresolved'])

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
            pot_events.at[i, 'overlap_hrs'] = int(round(overlap_hrs))

    @staticmethod
    def _compute_prior_to_claim(date_claim, pot_events):
        date_claim_end_day = datetime.combine(
            date_claim, datetime.max.time())
        for i, event in pot_events.iterrows():
            if event['e_start'] < date_claim_end_day:
                pot_events.at[i, 'prior'] = 1

    @staticmethod
    def _group_events_by_cid(events_df, sort_field):
        """
        Group the events per cell, sorted by the given date field. Built once
        before the matching loop so that the per-claim candidate lookup is a
        binary search within the cell instead of a scan of all events.
        """
        return {cid: group.sort_values(sort_field)
                for cid, group in events_df.groupby('cid', sort=False)}

    @staticmethod
    def _get_potential_classic_events(claim, events_by_cid, window_days,
                                      max_duration):
        """
        Get all potential events based on the CID and the date. The events must
        be grouped per cell and sorted by e_start (see _group_events_by_cid);
        window_days must be sorted in decreasing order; max_duration is the
        longest event duration (bounds the e_start search range).
        """
        cid_events = events_by_cid.get(claim['cid'])
        if cid_events is None:
            return None

        date_claim = claim['date_claim']

        # Define the starting and ending dates of the longest temporal window
        date_window_end, date_window_start = Damages._get_window_dates(
            date_claim, window_days[0])

        # Select all events overlapping the longest temporal window: their
        # e_start lies in [window start - longest duration, window end).
        starts = cid_events['e_start'].to_numpy()
        i0 = np.searchsorted(
            starts, pd.Timestamp(date_window_start - max_duration).to_datetime64())
        i1 = np.searchsorted(
            starts, pd.Timestamp(date_window_end).to_datetime64())
        potential_events = cid_events.iloc[i0:i1]
        potential_events = potential_events[
            potential_events['e_end'] > date_window_start]

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
    def _get_potential_simple_events(claim, events_by_cid):
        """
        Get all potential events based on the CID and the date. The events must
        be grouped per cell and sorted by i_max_date (see _group_events_by_cid).
        """
        cid_events = events_by_cid.get(claim['cid'])
        if cid_events is None:
            return None

        date_claim = claim['date_claim']

        # Define the starting and ending dates of the temporal window
        date_window_start = date_claim - timedelta(hours=SIMPLE_EVENT_HOURS_BEFORE)
        date_window_end = date_claim + timedelta(hours=SIMPLE_EVENT_HOURS_AFTER)

        # Select all events with i_max_date in the window (inclusive bounds)
        dates = cid_events['i_max_date'].to_numpy()
        i0 = np.searchsorted(dates, pd.Timestamp(date_window_start).to_datetime64())
        i1 = np.searchsorted(dates, pd.Timestamp(date_window_end).to_datetime64(),
                             side='right')

        if i1 <= i0:
            return None

        return cid_events.iloc[i0:i1].copy()

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

    def _store_in_claims_dataframe(self, df_claims):
        """
        Stores the claims for a given category in the dataframe.
        """
        self.claims = pd.merge(self.claims, df_claims, how='outer',
                               on=['date_claim', 'mask_index'], validate='one_to_one')

    def _extract_claims_from_grids(self, data, dates, category):
        """
        Vectorized extraction of the non-null claims of a (time, y, x) stack
        into a dataframe with columns [date_claim, mask_index, category].
        One np.nonzero call replaces the per-date _extract_non_null_claims loop.

        Parameters
        ----------
        data: np.ndarray
            The claim grids, shape (time, y, x).
        dates: list
            The dates (datetime.date) of the time axis.
        category: str
            The claim category (name of the value column).

        Returns
        -------
        pd.DataFrame
            The non-null claims.
        """
        assert data.ndim == 3, "Data should be 3D in _extract_claims_from_grids()."
        if self.mask['mask'].size == 0:
            raise RuntimeError("The mask for extraction was not defined.")

        masked = data[:, self.mask['mask']]  # (time, n_masked_cells)
        i_time, i_cell = np.nonzero(masked)

        if i_time.size == 0:
            df_claims = pd.DataFrame(
                columns=['date_claim', 'mask_index', category]).astype('int32')
            df_claims['date_claim'] = pd.to_datetime(df_claims['date_claim'])
            return df_claims

        return pd.DataFrame({
            'date_claim': np.asarray(dates, dtype=object)[i_time],
            'mask_index': i_cell.astype('int32'),
            category: np.asarray(masked[i_time, i_cell]),
        })

    def _extract_non_null_claims(self, data):
        """
        Extracts the cells with at least 1 claim.
        """
        # Extract the pixels where the catalog is not null
        assert data.ndim == 2, f"Data should be 2D in _extract_non_null_claims()."
        extracted = np.extract(self.mask['mask'], data[:, :])

        # Get non null data
        indices = np.nonzero(extracted)[0]
        values = extracted[indices]
        return indices, values

    def _check_extent(self, dataset, file):
        """
        Check extent consistency with other files.
        """
        if self.mask['extent'] == (None, None, None, None):
            if isinstance(dataset, rasterio.DatasetReader):
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
            elif isinstance(dataset, nc4.Dataset):
                self.mask['extent'] = (
                    dataset.variables['x'][:].min(),
                    dataset.variables['y'][:].min(),
                    dataset.variables['x'][:].max(),
                    dataset.variables['y'][:].max())
                # Extract the axes
                xs = dataset.variables['x'][:]
                ys = dataset.variables['y'][:]
                ys = ys.reshape(-1, 1)
                self.mask['xs'] = np.tile(xs, (len(ys), 1))
                self.mask['ys'] = np.tile(ys, (1, len(xs)))
            else:
                raise RuntimeError(f"Unknown dataset type for {file}.")

        else:
            if isinstance(dataset, rasterio.DatasetReader):
                if self.mask['extent'] != dataset.bounds:
                    raise RuntimeError(f"The extent of {file} differs from others.")
            elif isinstance(dataset, nc4.Dataset):
                if self.mask['extent'] != (
                        dataset.variables['x'][:].min(),
                        dataset.variables['y'][:].min(),
                        dataset.variables['x'][:].max(),
                        dataset.variables['y'][:].max()):
                    raise RuntimeError(f"The extent of {file} differs from others.")

    def _check_shape(self, data, file):
        """
        Check shape consistency with other files.
        """
        assert data.ndim == 2, f"Data should be 2D in _check_shape()."
        if self.mask['shape'] is None:
            self.mask['shape'] = data.shape
        elif self.mask['shape'] != data.shape:
            raise RuntimeError(f"The shape of {file} differs from other files.")

    def _create_mask(self, exposure_data):
        """
        Creates a mask with True for all pixels containing at least 1 annual exposure.
        """
        self.mask['mask'] = np.zeros(self.mask['shape'][:], dtype=bool)
        for arr in exposure_data:
            max_value = arr.max(axis=0)
            self.mask['mask'][max_value > 0] = True

    def _create_cids_list(self):
        """
        Creates the CIDs list for cells where we have exposure
        """
        xs_mask = np.extract(self.mask['mask'], self.mask['xs'])
        ys_mask = np.extract(self.mask['mask'], self.mask['ys'])

        xs_cid = self.domain.cids['xs'][0, :]
        ys_cid = self.domain.cids['ys'][:, 0]

        # Vectorized exact-match lookup of the coordinates on the CID axes
        x_idx = self._match_axis_indices(xs_cid, xs_mask, 'x')
        y_idx = self._match_axis_indices(ys_cid, ys_mask, 'y')

        self.cids_list = self.domain.cids['ids_map'][y_idx, x_idx].astype(float)

    @staticmethod
    def _nearest_axis_indices(axis, values):
        """
        Vectorized equivalent of argmin(|axis - v|) for each value, on a
        monotonic (ascending or descending) axis.
        """
        axis = np.asarray(axis, dtype='float64')
        values = np.asarray(values, dtype='float64')
        ascending = axis[0] <= axis[-1]
        axis_asc = axis if ascending else axis[::-1]
        pos = np.clip(np.searchsorted(axis_asc, values), 1, len(axis_asc) - 1)
        take_left = (np.abs(values - axis_asc[pos - 1])
                     <= np.abs(values - axis_asc[pos]))
        idx = np.where(take_left, pos - 1, pos)
        return idx if ascending else len(axis) - 1 - idx

    @staticmethod
    def _match_axis_indices(axis, values, what):
        """
        Vectorized index lookup of exact coordinate values on a monotonic axis.
        """
        axis = np.asarray(axis)
        values = np.asarray(values)
        ascending = axis[0] <= axis[-1]
        axis_asc = axis if ascending else axis[::-1]
        pos = np.searchsorted(axis_asc, values)
        pos = np.clip(pos, 0, len(axis_asc) - 1)
        matched = axis_asc[pos] == values
        if not matched.all():
            bad = values[np.argmax(~matched)]
            raise RuntimeError(f"No CID found for coordinate {what}={bad}.")
        return pos if ascending else len(axis) - 1 - pos

    def _extract_data_with_mask(self, data):
        """
        Extracts data according to the mask and returns a 2-D array.
        """
        if self.mask['mask'].size == 0:
            raise RuntimeError("The mask for extraction was not defined.")
        extracted = np.zeros((data.shape[0], np.sum(self.mask['mask'])), dtype=np.int16)
        for i in range(data.shape[0]):
            extracted[i, :] = np.extract(self.mask['mask'], data[i, :, :])
        return extracted

    def _load_from_dump(self, filename):
        """
        Loads the object content from a pickle file.
        """
        if not self.use_dump:
            return
        file_path = Path(self.pickles_dir + '/' + filename)
        if not file_path.is_file():
            return
        with open(file_path, 'rb') as f:
            values = pickle.load(f)
            self.mask = values.mask
            self.exposure = values.exposure
            self.claims = values.claims
            self.cids_list = values.cids_list
            if hasattr(values, 'selected_claim_categories'):
                self.selected_claim_categories = values.selected_claim_categories
            if hasattr(values, 'selected_exposure_categories'):
                self.selected_exposure_categories = values.selected_exposure_categories

    def _dump_object(self, filename=None):
        """
        Saves the object content to a pickle file.
        """
        if not self.use_dump:
            return
        if filename is None:
            filename = f'damages_{self.name}_{self.year_start}-{self.year_end}.pickle'
        file_path = Path(self.pickles_dir + '/' + filename)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def _initialize_exposure_dataframe(self, exposure_data_cat):
        """
        Initializes the exposure dataframe by filling the year and the mask_index columns.
        The mask_index column refers to the 1-D array after extraction by the mask.
        """
        n_years = self.year_end - self.year_start + 1
        n_annual_rows = exposure_data_cat.shape[1]
        years = np.repeat(np.arange(self.year_start, self.year_end + 1), n_annual_rows)
        self.exposure['year'] = years
        indices = np.tile(np.arange(n_annual_rows), n_years)
        self.exposure['mask_index'] = indices

    def _set_to_exposure_dataframe(self, exposure_data_cat, category):
        """
        Sets the exposure data to the dataframe for the given category.
        """
        exposure = np.reshape(exposure_data_cat, exposure_data_cat.size)
        self.exposure[category] = exposure

    def _clean_claims_dataframe(self):
        """
        Reorder claims dataframe and remove nans.
        """
        columns = ['date_claim', 'mask_index', 'selection'] + self.claim_categories
        self.claims = self.claims.reindex(columns=columns)
        self.claims.fillna(0, inplace=True)
        self.claims.sort_values(by=['date_claim', 'mask_index'], inplace=True)
        self.claims.reset_index(inplace=True, drop=True)
        for category in self.claim_categories:
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

    def _set_exposure_cids(self):
        xs_mask_extracted = np.extract(self.mask['mask'], self.mask['xs'])
        ys_mask_extracted = np.extract(self.mask['mask'], self.mask['ys'])
        cids = self.cids_list[self.exposure['mask_index']].astype(np.int32)
        x = xs_mask_extracted[self.exposure['mask_index']].astype(np.int32)
        y = ys_mask_extracted[self.exposure['mask_index']].astype(np.int32)
        self.exposure.insert(2, 'cid', cids)
        self.exposure.insert(3, 'x', x)
        self.exposure.insert(4, 'y', y)

        # Remove rows with cid = nan or 0
        self.exposure = self.exposure[self.exposure.cid.notnull()]
        self.exposure = self.exposure[self.exposure.cid != 0]

    def _extract_exposure_data(self, directory):
        raise NotImplementedError("This method should be implemented in a child class.")

    def _extract_claim_data(self, directory):
        raise NotImplementedError("This method should be implemented in a child class.")
