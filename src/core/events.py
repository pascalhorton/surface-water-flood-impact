"""
Class to handle the events.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.config import Config

config = Config()


class Events:
    def __init__(self, use_dump=True):
        """
        The Events class.

        Parameters
        ----------
        use_dump: bool
            Dump the content to the TMP_DIR and load if available
        """
        self.use_dump = use_dump
        self.events = None

        self._load_from_dump()

    def load_events_and_select_locations_with_contracts(self, path, damages):
        """
        Load all events from a parquet file. Then, select only the events where there
        is a contract.

        Parameters
        ----------
        path: str
            The path to the parquet file.
        damages: Damages instance
            The damages object containing the contracts and claims data.
        """
        if self.use_dump and self.events is not None:
            print("Events were reloaded from pickle file.")
            return

        self.events = pd.read_parquet(path)
        print("Events were loaded from parquet file.")
        print(f"Number of all events: {len(self.events)}")

        self.select_locations_with_contracts(damages)
        self._add_event_id()
        self._dump_object()

    def get_events_sample(self):
        """
        Get a small sample of the events dataframe.

        Returns
        -------
        The first 100 rows of the events dataframe.
        """
        return self.events[0:100]

    def select_locations_with_contracts(self, damages):
        """
        Select only the events where there is a contract.

        Parameters
        ----------
        damages: Damages instance
            The damages object containing the contracts and claims data.
        """
        cids = damages.cids_list
        self.events = self.events[self.events['cid'].isin(cids)]
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
        target_values = damages.claims.loc[:, ['eid', 'target']]

        # Merge the target values with the events
        self.events = pd.merge(self.events, target_values,
                               how="left", on=['eid'])
        self.events['target'] = self.events['target'].fillna(0)

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

    def load_from_pickle(self, filename='events.pickle'):
        """
        Load the events from a pickle file.

        Parameters
        ----------
        filename: str
            The filename of the pickle file.
        """
        self.use_dump = True
        self._load_from_dump(filename)

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
        tmp_dir = config.get('TMP_DIR')
        file_path = Path(f'{tmp_dir}/{filename}')
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
        tmp_dir = config.get('TMP_DIR')
        file_path = Path(f'{tmp_dir}/{filename}')
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def _add_event_id(self):
        """
        Add an incremental event ID
        """
        ids = np.arange(start=1, stop=len(self.events) + 1)
        self.events.insert(0, 'eid', ids)
