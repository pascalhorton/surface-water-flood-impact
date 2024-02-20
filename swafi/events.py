"""
Class to handle the events.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

config = Config()


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
        self.events = self.events[
            (self.events['e_start'].dt.year >= damages.year_start) &
            (self.events['e_start'].dt.year <= damages.year_end)
            ]

        print(f"Number of events with contracts in "
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

        print(f"Number of events with contracts: {len(self.events)}")

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
        self.events = self.events[
            (self.events['e_end'] < start_date) |
            (self.events['e_end'] > end_date)
            ]

    def count_positives(self):
        """
        Count the number of positive events.

        Returns
        -------
        The number of positive targets.
        """
        return (self.events['target'] > 0).sum()

    def reduce_number_of_negatives(self, nb_keep, random_state=42):
        """
        Reduce the number of events with target = 0.

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

        # Sort the events by eid
        self.events = self.events.sort_values(by=['eid'])

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
        self.events['year'] = pd.to_datetime(self.events['e_start']).dt.year
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
        if file_path.is_file():
            with open(file_path, 'rb') as f:
                values = pickle.load(f)
                self.events = values.events

    def _dump_object(self, filename='events.pickle'):
        """
        Saves the object content to a pickle file.
        """
        if not self.use_dump:
            return
        pickles_dir = config.get('PICKLES_DIR')
        file_path = Path(f'{pickles_dir}/{filename}')
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

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
    if not file_path.is_file():
        raise Exception(f"File {file_path} does not exist.")

    events = Events(use_dump=False)
    with open(file_path, 'rb') as f:
        values = pickle.load(f)
        events.events = values.events

        # Check that there is no event without contract
        assert not (events.events['nb_contracts'] == 0).any()

    return events
