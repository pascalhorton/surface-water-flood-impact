"""
Class to handle the events.
"""

import pickle
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from .config import Config

config = Config()


_PICKLE_LOAD_EXCEPTIONS = (
    OSError,
    EOFError,
    pickle.UnpicklingError,
    AttributeError,
    ValueError,
    TypeError,
    NotImplementedError,
)


class Events:
    def __init__(self, use_dump=True):
        """
        The Events class.

        Parameters
        ----------
        use_dump: bool
            Dump the content to the PICKLES_DIR and load if available
        """
        self.use_dump = use_dump
        self.events = None

        self._load_from_dump()

    def load_events_and_select_those_with_contracts(self, path, damages, tag):
        """
        Load all events from a parquet file. Then, select only the events where there
        is a contract.

        Parameters
        ----------
        path: str
            The path to the parquet file.
        damages: Damages instance
            The damages object containing the contracts and claims data.
        tag: str
            The tag to add to the pickle file (e.g. damage dataset).
        """
        if self.use_dump and self.events is not None:
            print("Events were reloaded from pickle file.")
            return

        self.events = pd.read_parquet(path)
        print("Events were loaded from parquet file.")
        print(f"Number of all events: {len(self.events)}")

        self.select_years_with_contracts(damages)
        self.select_locations_with_contracts(damages)
        self._add_event_id()
        self._dump_object(f'events_{tag}.pickle')

    def get_events_sample(self):
        """
        Get a small sample of the events dataframe.

        Returns
        -------
        The first 100 rows of the events dataframe.
        """
        return self.events[0:100]

    def select_years_with_contracts(self, damages):
        """
        Select only the years with a contract.

        Parameters
        ----------
        damages: Damages instance
            The damages object containing the contracts and claims data.
        """
        date_field = ''
        if 'e_start' in self.events.columns:
            date_field = 'e_start'
        elif 'e_date' in self.events.columns:
            date_field = 'e_date'
        elif 'date' in self.events.columns:
            date_field = 'date'
        else:
            raise ValueError("No date field found in damages claims.")

        self.events = self.events[
            (self.events[date_field].dt.year >= damages.year_start) &
            (self.events[date_field].dt.year <= damages.year_end)
            ]

        print(f"Number of events with potential contracts in "
              f"the selected years: {len(self.events)}")

    def select_locations_with_contracts(self, damages):
        """
        Select only the events where there is a contract.

        Parameters
        ----------
        damages: Damages instance
            The damages object containing the contracts and claims data.
        """

        # First, select the events where there is a contract in any year
        cids = damages.cids_list
        self.events = self.events[self.events['cid'].isin(cids)]

        # Second, remove cells where there is no annual contract
        empty_cells = damages.exposure[damages.exposure['selection'] == 0]
        for index, row in empty_cells.iterrows():
            cid = cids[row['mask_index']]
            self.events = self.events[
                (self.events['cid'] != cid) |
                (self.events['e_start'].dt.year != row['year'])
                ]

        print(f"Number of events with potential contracts: {len(self.events)}")

    def set_target_values_from_damages(self, damages):
        """
        Set the target values for the events from the damage data. The damage data
        must be linked to the events.

        Parameters
        ----------
        damages: Damages instance
            An object containing the damages properties.
        """
        target_values = damages.claims.loc[:, ['date_claim', 'eid',
                                               'selection', 'target']]

        # Rename the column selection to nb_claims
        target_values.rename(columns={'selection': 'nb_claims'}, inplace=True)

        # Merge the target values with the events
        self.events = pd.merge(self.events, target_values,
                               how="left", on=['eid'])
        self.events['target'] = self.events['target'].fillna(0)
        self.events['nb_claims'] = self.events['nb_claims'].fillna(0)

    def get_events_for_removed_claims(self, removed_claims, damages):
        """
        Get the events for the removed claims.

        Parameters
        ----------
        removed_claims: pd.DataFrame
            The removed claims.
        damages: Damages instance
            An object containing the damages properties.

        Returns
        -------
        The events for the removed claims.
        """
        print("Extracting events for the removed claims.")

        cids = removed_claims['cid'].unique()

        # Select the event cids for the removed claims
        events = self.events.copy()
        events = events[events['cid'].isin(cids)]

        # Compute the middle-date of the events
        if 'e_date' in events.columns:
            events['mid_date'] = events['e_date']
            n_days = 1
        else:
            events['mid_date'] = events['e_start'] + (events['e_end'] - events['e_start']) / 2
            n_days = 2

        events_to_remove = []
        for i_claim in tqdm(range(len(removed_claims)), desc=f"Checking events"):
            claim = removed_claims.iloc[i_claim]
            mask = (events['cid'] == claim['cid']) & \
                   (events['mid_date'] >= claim['date_claim'] - pd.Timedelta(days=n_days)) & \
                   (events['mid_date'] <= claim['date_claim'] + pd.Timedelta(days=n_days))
            events_to_remove.extend(events.loc[mask, 'eid'].tolist())

        # Filter out the events that are associated with damages
        events_to_remove = [ev for ev in events_to_remove if
                            ev not in damages.claims.eid.tolist()]

        print(f"Events to remove dues to claim classes: {len(events_to_remove)}")

        return events_to_remove

    def remove_period(self, start_date, end_date):
        """
        Remove a period of time from the events.

        Parameters
        ----------
        start_date: str
            The start date of the period to remove.
        end_date: str
            The end date of the period to remove.
        """
        if 'e_end' in self.events.columns:
            date_field = 'e_end'
        elif 'e_date' in self.events.columns:
            date_field = 'e_date'
        else:
            raise ValueError("No date field found in events.")

        self.events = self.events[
            (self.events[date_field] < start_date) |
            (self.events[date_field] > end_date)
            ]

    def remove_events(self, events_to_remove):
        """
        Remove specific events from the events dataframe.

        Parameters
        ----------
        events_to_remove: list
            A list of event IDs to remove.
        """
        self.events = self.events[~self.events['eid'].isin(events_to_remove)]

    def remove_events_without_contracts(self):
        """
        Remove events without contracts.
        """
        len_before = len(self.events)
        self.events.dropna(subset=['nb_contracts'], inplace=True)
        len_after = len(self.events)
        print(f"Number of events without actual contracts: {len_before - len_after}")

    def count_positives(self):
        """
        Count the number of positive events.

        Returns
        -------
        The number of positive targets.
        """
        return (self.events['target'] > 0).sum()

    def reduce_number_of_negatives(self, nb_keep, random_state=None):
        """
        Reduce the number of events with target = 0.
        Not used in the current version!

        Parameters
        ----------
        nb_keep: int
            The number of events to keep.
        random_state: int
            The random state.
        """
        print("Reducing the number of negative events.")
        print(f"Number of events before reduction: {len(self.events)}")

        # Select only the negative events
        negatives = self.events[self.events['target'] == 0]

        # Reduce the number of negative events
        negatives = negatives.sample(n=nb_keep, random_state=random_state)

        # Merge the negative and positive events
        positives = self.events[self.events['target'] > 0]
        self.events = pd.concat([positives, negatives])

        # Shuffle the events
        self.events = self.events.sample(frac=1, random_state=random_state).reset_index(drop=True)

        print(f"Number of events after reduction: {len(self.events)}")

    def set_contracts_number(self, damages):
        """
        Set the number of contracts per cell.

        Parameters
        ----------
        damages: Damages instance
            An object containing the damages properties.
        """
        contracts_number = damages.exposure[['cid', 'year', 'selection']].copy()
        contracts_number.rename(columns={'selection': 'nb_contracts'}, inplace=True)

        # Merge the target values with the events
        if 'e_start' in self.events.columns:
            self.events['year'] = pd.to_datetime(self.events['e_start']).dt.year
        else:
            self.events['year'] = pd.to_datetime(self.events['e_date']).dt.year
        self.events = pd.merge(self.events, contracts_number,
                               how="left", on=['cid', 'year'])

    def save_to_pickle(self, filename='events.pickle'):
        """
        Save the events to a pickle file.

        Parameters
        ----------
        filename: str
            The filename of the pickle file.
        """
        self.use_dump = True
        self._dump_object(filename)

    def save_to_csv(self, filename='events.csv'):
        """
        Save the events to a csv file.

        Parameters
        ----------
        filename: str
            The filename of the csv file.
        """
        tmp_dir = config.get('TMP_DIR')
        file_path = Path(f'{tmp_dir}/{filename}')
        self.events.to_csv(file_path, index=False)

    def _load_from_dump(self, filename='events.pickle'):
        """
        Loads the object content from a pickle file.
        """
        if not self.use_dump:
            return
        pickles_dir = config.get('PICKLES_DIR')
        file_path = Path(f'{pickles_dir}/{filename}')

        events_only_path = _events_only_pickle_path(pickles_dir, filename)

        # Try to load from the main pickle first (Events object or DataFrame)
        if file_path.is_file():
            try:
                self.events = _load_events_from_file(file_path)
                return
            except (_PICKLE_LOAD_EXCEPTIONS, TypeError):
                # If full-object unpickling fails, fall back to events-only file
                pass

        # Fallback: try to load the events-only gzipped pickle
        if events_only_path.is_file():
            self.events = pd.read_pickle(events_only_path, compression='gzip')

    def _dump_object(self, filename='events.pickle'):
        """
        Saves the object content to a pickle file. If pickling the whole object fails
        (commonly on Windows for very large objects), fall back to saving only the
        events DataFrame compressed with gzip.
        """
        if not self.use_dump:
            return
        pickles_dir = config.get('PICKLES_DIR')
        file_path = Path(f'{pickles_dir}/{filename}')

        events_only_path = _events_only_pickle_path(pickles_dir, filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        events_only_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Attempt to pickle the whole Events instance
            data_bytes = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
            file_path.write_bytes(data_bytes)
        except (OSError, OverflowError, pickle.PicklingError, MemoryError):
            if self.events is None:
                raise

        # Always persist a portable events-only file to avoid cross-version pickle issues
        if self.events is None:
            raise ValueError("Cannot dump events because self.events is None.")
        self.events.to_pickle(events_only_path, compression='gzip')

    def _add_event_id(self):
        """
        Add an incremental event ID
        """
        ids = np.arange(start=1, stop=len(self.events) + 1)
        self.events.insert(0, 'eid', ids)


def load_events_from_pickle(filename='events.pickle'):
    """
    Load the events from a pickle file.

    Parameters
    ----------
    filename: str
        The filename of the pickle file.
    """
    pickles_dir = config.get('PICKLES_DIR')
    file_path = Path(f'{pickles_dir}/{filename}')
    events_only_path = _events_only_pickle_path(pickles_dir, filename)

    events = Events(use_dump=False)

    # Try to load from the main pickle first (Events object or DataFrame)
    if file_path.is_file():
        try:
            events.events = _load_events_from_file(file_path)
        except (_PICKLE_LOAD_EXCEPTIONS, TypeError):
            # Fallback to events-only gzipped pickle
            if not events_only_path.is_file():
                raise Exception(
                    f"Failed to load {file_path}. Fallback file {events_only_path} "
                    f"does not exist or could not be unpickled."
                )
            events.events = pd.read_pickle(events_only_path, compression='gzip')
    else:
        if not events_only_path.is_file():
            raise Exception(f"File {file_path} or {events_only_path} does not exist.")
        events.events = pd.read_pickle(events_only_path, compression='gzip')

    # Check that there is no event without contract
    if 'nb_contracts' not in events.events.columns:
        raise AssertionError("Loaded events do not contain 'nb_contracts' column.")
    if events.events['nb_contracts'].eq(0).any():
        raise AssertionError("There are events without contracts (nb_contracts == 0).")

    return events


def _events_only_pickle_path(pickles_dir, filename):
    """Return the portable events-only pickle path for a given filename."""
    return Path(f'{pickles_dir}/{Path(filename).stem}_events.pkl.gz')


def _extract_events_dataframe(loaded_value):
    """Normalize objects loaded from pickle into an events DataFrame."""
    if isinstance(loaded_value, pd.DataFrame):
        return loaded_value
    if hasattr(loaded_value, 'events'):
        return loaded_value.events
    raise TypeError(f"Unsupported pickled object type: {type(loaded_value)!r}")


def _load_events_from_file(file_path):
    """Load events from a pickle path using stdlib pickle then pandas fallback."""
    try:
        with open(file_path, 'rb') as f:
            loaded_value = pickle.load(f)
    except _PICKLE_LOAD_EXCEPTIONS:
        loaded_value = pd.read_pickle(file_path)
    return _extract_events_dataframe(loaded_value)

