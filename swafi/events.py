"""
Class to handle the events.
"""

import logging
import pickle
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from .config import Config
from .utils.event_extraction import detection_tag_in

config = Config()

logger = logging.getLogger(__name__)


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
        # Reload the tagged dump (never a generic one: another run's events
        # would silently be reused for the wrong dataset/method). The tag
        # carries the damage dataset, the method and the precipitation
        # dataset - and NOT how the events were detected, which is recorded
        # only in the name of the parquet. So the detection tag is appended:
        # without it, re-extracting a catalogue with new detection settings
        # and re-running the linkage reloads the previous events from this
        # dump and never opens the new parquet at all. The result is written
        # under the new name while holding the old catalogue, and nothing
        # downstream can tell.
        dump_name = f'events_{tag}{detection_tag_in(path)}.pickle'
        if self.use_dump and self.events is None:
            self._load_from_dump(dump_name)
        if self.use_dump and self.events is not None:
            logger.info("Events were reloaded from the pickle file '%s' "
                        "rather than read from '%s'. Delete it to force a "
                        "re-read.", dump_name, Path(path).name)
            return

        self.events = pd.read_parquet(path)
        logger.info("Events were loaded from parquet file.")
        logger.info("Number of all events: %s", len(self.events))

        self.select_years_with_contracts(damages)
        self.select_locations_with_contracts(damages)
        self._add_event_id()
        self._dump_object(dump_name)

    def check_precip_dataset(self, expected):
        """
        Check that the loaded events were extracted from the expected
        precipitation dataset (provenance column stamped at extraction).

        Parameters
        ----------
        expected: str
            The expected precipitation dataset (e.g. 'hourly' or '5min').
        """
        if 'precip_dataset' not in self.events.columns:
            logger.warning("The events carry no 'precip_dataset' column (legacy "
                           "file): cannot verify they are '%s' events.", expected)
            return

        found = set(self.events['precip_dataset'].unique().tolist())
        if found != {expected}:
            raise ValueError(
                f"The loaded events were extracted from the {sorted(found)} "
                f"precipitation dataset(s), but '{expected}' is expected. "
                f"Check the events path / pickle tag.")
        logger.info("Events provenance check passed: '%s' precipitation.", expected)

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
            raise ValueError("No date field found in events.")

        self.events = self.events[
            (self.events[date_field].dt.year >= damages.year_start) &
            (self.events[date_field].dt.year <= damages.year_end)
            ]

        logger.info("Number of events with potential contracts in the selected years: %s", len(self.events))

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

        # Second, keep only the (cid, year) pairs that actually have an annual
        # contract. Expressed as a positive semi-join on (cid, year): the former
        # anti-join on `selection == 0` was a no-op because the exposure is
        # already filtered to `selection != 0` upstream, so those rows never
        # existed. The semi-join stays correct whatever the exposure state.
        if 'e_start' in self.events.columns:
            date_field = 'e_start'
        elif 'e_date' in self.events.columns:
            date_field = 'e_date'
        else:
            raise ValueError("No date field found in events.")

        valid_cells = damages.exposure[damages.exposure['selection'] > 0]
        valid_pairs = pd.MultiIndex.from_arrays(
            [valid_cells['cid'].astype('float64'),
             valid_cells['year'].to_numpy()])
        event_pairs = pd.MultiIndex.from_arrays(
            [self.events['cid'].astype('float64'),
             self.events[date_field].dt.year])
        self.events = self.events[event_pairs.isin(valid_pairs)]

        logger.info("Number of events with potential contracts: %s", len(self.events))

    def set_target_values_from_damages(self, damages):
        """
        Set the target values for the events from the damage data. The damage data
        must be linked to the events.

        Parameters
        ----------
        damages: Damages instance
            An object containing the damages properties.
        """
        cols = ['date_claim', 'eid', 'selection', 'target']
        # Carry the matching stage through when the linkage recorded one, so a
        # training run can drop the relaxed positives without redoing the
        # linkage. It is NOT a feature and must never become one: it is zero for
        # every negative, so a model given it would score perfectly and learn
        # nothing. Feature selection is explicit ('class:feature'), which is
        # what keeps that from happening by accident.
        has_step = 'match_step' in damages.claims.columns
        if has_step:
            cols.append('match_step')
        target_values = damages.claims.loc[:, cols]
        target_values = target_values.rename(columns={'selection': 'nb_claims'})

        # Aggregate per event: several claims can be linked to the same event
        # (same eid). Collapsing them here keeps the merge one-to-one, so an
        # event is never duplicated in the output. The 'simple' method also
        # relies on remove_duplicates() (same cid/i_max_date), but the 'classic'
        # method had no such step and produced duplicate event rows. nb_claims
        # are summed; the target is summed then clipped to 1 — all claims of an
        # event share the same nb_contracts, so summed damage ratios give the
        # event's total ratio, and summed occurrence flags collapse to 1.
        agg = dict(date_claim=('date_claim', 'min'),
                   nb_claims=('nb_claims', 'sum'),
                   target=('target', 'sum'))
        if has_step:
            # MIN, not max: one event can be the strict match of one claim and
            # the relaxed match of another, and such an event would be a
            # positive with the relaxed stage switched off. Taking the minimum
            # marks it strict, so dropping match_step == 2 removes only the
            # positives that exist BECAUSE of the relaxed stage.
            agg['match_step'] = ('match_step', 'min')
        target_values = target_values.groupby('eid', as_index=False).agg(**agg)
        target_values['target'] = target_values['target'].clip(upper=1)

        # Merge the target values with the events (one event per eid)
        self.events = pd.merge(self.events, target_values,
                               how="left", on=['eid'], validate='m:1')
        self.events['target'] = self.events['target'].fillna(0)
        self.events['nb_claims'] = self.events['nb_claims'].fillna(0)
        if has_step:
            self.events['match_step'] = self.events['match_step'].fillna(0)

    def drop_relaxed_positives(self):
        """Demote the positives that exist only because of the relaxed stage.

        The relaxed second stage of the claim linkage matches a claim to a
        storm one day further back than the strict window allows, on the
        reasoning that the claim date is wrong rather than the storm absent.
        The placebo measurement puts that at roughly even odds (see the
        constants in swafi/damages.py), so whether those positives help or
        hurt is an empirical question - and this is the switch that asks it.

        The target is ZEROED rather than the row dropped, because that is what
        the strict linkage produces: a claim the strict window cannot match is
        discarded, while the storm it would have been matched to stays in the
        catalogue as a negative. Dropping the row would remove a candidate the
        strict run keeps, and the two samples would then differ by more than
        their labels.

        One residual difference from a true strict run cannot be undone here:
        where a stage-2 claim had several candidates in the extension band,
        the linkage removed the ones it did not pick, and a strict run would
        have kept them all as negatives. Measured 2026-10-26 by linking gvz
        hourly both ways against the same catalogue: 1,163,776 events relaxed
        against 1,163,805 strict, so the two samples differ by **29 rows** -
        0.0025% of the sample - on top of the 103 labels this method flips.
        The demoted sample is therefore a faithful stand-in for a strict one.

        Returns
        -------
        int
            The number of positives demoted.
        """
        if 'match_step' not in self.events.columns:
            raise ValueError(
                "This events file has no 'match_step' column, so it cannot "
                "say which positives came from the relaxed stage. Either it "
                "was linked before the two-stage matching existed, or it was "
                "linked with relaxed matching off - in which case there is "
                "nothing to drop and the option should not be set.")

        n_pos = int((self.events['target'] > 0).sum())
        is_relaxed = self.events['match_step'] == 2
        n = int(is_relaxed.sum())

        # nb_claims goes to zero too: Impact keeps rows with nb_claims == 0
        # unconditionally (the min_nb_claims filter spares them explicitly),
        # so a demoted event stays in the sample as a negative rather than
        # being filtered out of it. match_step follows, keeping the invariant
        # that 0 means 'not a positive'.
        self.events.loc[is_relaxed, 'target'] = 0
        self.events.loc[is_relaxed, 'nb_claims'] = 0
        self.events.loc[is_relaxed, 'match_step'] = 0

        logger.info(
            "Relaxed positives demoted to negatives: %s of %s positives "
            "(%.1f%%); %s positives remain.",
            n, n_pos, 100 * n / n_pos if n_pos else 0.0, n_pos - n)
        return n

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
        logger.info("Extracting events for the removed claims.")

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

        # Group the events per cell (sorted by date): the per-claim lookup is
        # then a binary search instead of a scan of the whole dataframe.
        events_by_cid = {cid: group.sort_values('mid_date')
                         for cid, group in events.groupby('cid', sort=False)}

        events_to_remove = []
        for i_claim in tqdm(range(len(removed_claims)), desc=f"Checking events"):
            claim = removed_claims.iloc[i_claim]
            cid_events = events_by_cid.get(claim['cid'])
            if cid_events is None:
                continue
            mid_dates = cid_events['mid_date'].to_numpy()
            i0 = np.searchsorted(
                mid_dates,
                (claim['date_claim'] - pd.Timedelta(days=n_days)).to_datetime64())
            i1 = np.searchsorted(
                mid_dates,
                (claim['date_claim'] + pd.Timedelta(days=n_days)).to_datetime64(),
                side='right')
            events_to_remove.extend(cid_events['eid'].to_numpy()[i0:i1].tolist())

        # Filter out the events that are associated with damages
        linked_eids = set(damages.claims['eid'].tolist())
        events_to_remove = [ev for ev in events_to_remove if
                            ev not in linked_eids]

        logger.info("Events to remove dues to claim classes: %s", len(events_to_remove))

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
        logger.info("Number of events without actual contracts: %s", len_before - len_after)

    def remove_duplicates(self):
        """
        Remove duplicate events with the same cid and i_max_date.
        The row with the highest nb_claims is kept; its nb_claims is set
        to the sum of all duplicates' nb_claims.
        """
        len_before = len(self.events)
        grouped = self.events.groupby(['cid', 'i_max_date'], sort=False)
        keep_idx = grouped['nb_claims'].idxmax()
        claims_sum = grouped['nb_claims'].transform('sum')
        # MIN over the POSITIVES of the group, for the same reason as in
        # set_target_values_from_damages: the kept row is the one with most
        # claims, which need not be the one the strict stage matched, while
        # the summed nb_claims covers the whole group. Taking the minimum
        # keeps a group holding any strict match marked strict, so dropping
        # match_step == 2 never removes a positive the strict linkage would
        # also have produced. The zeros are masked out first - a group of one
        # positive and several negatives must keep the positive's stage, not
        # collapse to 0, which means 'not a positive'.
        if 'match_step' in self.events.columns:
            step = self.events['match_step'].replace(0, np.nan)
            step_min = step.groupby(
                [self.events['cid'], self.events['i_max_date']],
                sort=False).transform('min')
            self.events['match_step'] = step_min.fillna(0)
        self.events['nb_claims'] = claims_sum
        self.events = self.events.loc[keep_idx].reset_index(drop=True)
        logger.info("Duplicate events removed: %s", len_before - len(self.events))

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
        logger.info("Reducing the number of negative events.")
        logger.info("Number of events before reduction: %s", len(self.events))

        # Select only the negative events
        negatives = self.events[self.events['target'] == 0]

        # Reduce the number of negative events
        negatives = negatives.sample(n=nb_keep, random_state=random_state)

        # Merge the negative and positive events
        positives = self.events[self.events['target'] > 0]
        self.events = pd.concat([positives, negatives])

        # Shuffle the events
        self.events = self.events.sample(frac=1, random_state=random_state).reset_index(drop=True)

        logger.info("Number of events after reduction: %s", len(self.events))

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
            except _PICKLE_LOAD_EXCEPTIONS:
                # If full-object unpickling fails, fall back to events-only file
                pass

        # Fallback: try to load the events-only pickle (bz2, with legacy gzip fallback)
        loaded = _read_events_only_pickle(events_only_path)
        if loaded is not None:
            self.events = loaded

    def _dump_object(self, filename='events.pickle'):
        """
        Saves the object content to a pickle file. If pickling the whole object fails
        (commonly on Windows for very large objects), fall back to saving only the
        events DataFrame compressed with bz2.
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
        try:
            self.events.to_pickle(events_only_path, compression='bz2')
        except Exception as e:
            logger.warning(
                "Could not save events-only pickle (%s: %s). "
                "Computations will continue but the file was not saved.",
                type(e).__name__, e,
            )

    def _add_event_id(self):
        """
        Add an incremental event ID
        """
        ids = np.arange(start=1, stop=len(self.events) + 1)
        self.events.insert(0, 'eid', ids)


def get_events_filename(dataset, event_file_label, event_method,
                        precip_dataset='hourly', extension='.pickle'):
    """
    Compose the name of the events file holding the target values.

    Single source of truth for that name: it is used both where the file is
    written (the claims-events linkage) and where it is read back (the training
    and assessment scripts). The name carries the extraction method and, for the
    simple method, the precipitation dataset the events were extracted from. The
    classic method relies on hourly data by definition and is not tagged.

    Parameters
    ----------
    dataset: str
        The damage dataset ('mobiliar' or 'gvz').
    event_file_label: str
        The event file label (e.g. 'default_occurrence').
    event_method: str
        The event extraction method ('simple' or 'classic').
    precip_dataset: str
        The precipitation dataset the events were extracted from ('hourly' or
        '5min'). Only tagged for the simple method.
    extension: str
        The file extension to append (e.g. '.pickle' or '.csv'). Pass '' to get
        the base name.

    Returns
    -------
    str
        The events filename.
    """
    if event_method not in ['simple', 'classic']:
        raise ValueError(
            f"Invalid event method: {event_method}. Expected 'simple' or "
            f"'classic' (use --event-method to set it).")
    if precip_dataset not in ['hourly', '5min']:
        raise ValueError(f"Unknown precipitation dataset: {precip_dataset}")
    if event_method == 'classic' and precip_dataset != 'hourly':
        raise ValueError("The classic method relies on hourly data.")

    precip_suffix = f'_{precip_dataset}' if event_method == 'simple' else ''

    return (f'events_{dataset}_with_target_{event_file_label}_'
            f'{event_method}{precip_suffix}{extension}')


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
        except _PICKLE_LOAD_EXCEPTIONS:
            # Fallback to events-only pickle (bz2, with legacy gzip fallback)
            loaded = _read_events_only_pickle(events_only_path)
            if loaded is None:
                raise Exception(
                    f"Failed to load {file_path}. Fallback file {events_only_path} "
                    f"does not exist or could not be unpickled."
                )
            events.events = loaded
    else:
        loaded = _read_events_only_pickle(events_only_path)
        if loaded is None:
            raise Exception(f"File {file_path} or {events_only_path} does not exist.")
        events.events = loaded

    # Check that there is no event without contract
    if 'nb_contracts' not in events.events.columns:
        raise AssertionError("Loaded events do not contain 'nb_contracts' column.")
    if events.events['nb_contracts'].eq(0).any():
        raise AssertionError("There are events without contracts (nb_contracts == 0).")

    return events


def _events_only_pickle_path(pickles_dir, filename):
    """Return the portable events-only pickle path for a given filename."""
    return Path(f'{pickles_dir}/{Path(filename).stem}_events.pkl.bz2')


def _read_events_only_pickle(events_only_path):
    """Read events-only pickle, falling back to legacy .pkl.gz if needed."""
    if events_only_path.is_file():
        return pd.read_pickle(events_only_path, compression='bz2')
    legacy_path = events_only_path.with_suffix('').with_suffix('.pkl.gz')
    if legacy_path.is_file():
        return pd.read_pickle(legacy_path, compression='gzip')
    return None


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

