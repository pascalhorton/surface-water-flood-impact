"""
This script computes the link between the claims and the events. The link is
computed using the following criteria:
    - i_mean: mean intensity of the event
    - i_max: max intensity of the event
    - p_sum: sum of the event precipitation
    - r_ts_win: ratio of the event time steps within the temporal window on the
      total window duration
    - r_ts_evt: ratio of the event time steps within the temporal window on the
      total event duration
    - prior: put more weights on events occurring prior to the claim
"""

import logging
import pickle
from swafi.config import Config
from swafi.damages_mobiliar import DamagesMobiliar
from swafi.damages_gvz import DamagesGvz
from swafi.events import Events, get_events_filename
from swafi.utils.logging_setup import setup_logging
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG = Config()

DATASET = 'gvz'  # 'mobiliar' or 'gvz'

# Events extraction method ('classic' for Bernet et al 2019 or 'simple' for the new
# simple approach). Must be the same as the one used for the events extraction
METHOD = 'simple'

# Precipitation dataset the events were extracted from ('hourly' or '5min');
# simple method only — the classic method relies on hourly data by definition.
PRECIP_DATASET = 'hourly'

# Only for the classic approach
CRITERIA = ['prior', 'i_mean', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt']
LABEL_DAMAGE_LINK = 'default'
WINDOW_DAYS = [5, 3, 1]

# Common options
PICKLES_DIR = CONFIG.get('PICKLES_DIR')
if METHOD == 'simple':
    if PRECIP_DATASET == 'hourly':
        EVENTS_PATH = CONFIG.get('EVENTS_PATH_SIMPLE_HOURLY')
    elif PRECIP_DATASET == '5min':
        EVENTS_PATH = CONFIG.get('EVENTS_PATH_SIMPLE_5MIN')
    else:
        raise ValueError(f"Unknown precipitation dataset: {PRECIP_DATASET}")
    # Every file name of the simple pipeline carries the dataset explicitly
    PRECIP_SUFFIX = f'_{PRECIP_DATASET}'
else:
    if PRECIP_DATASET != 'hourly':
        raise ValueError("The classic method relies on hourly data.")
    EVENTS_PATH = CONFIG.get('EVENTS_PATH_CLASSIC')
    PRECIP_SUFFIX = ''
EVENTS_TAG = f'{DATASET}_{METHOD}{PRECIP_SUFFIX}'
TARGET_TYPE = 'occurrence'  # 'occurrence' or 'damage_ratio'
# The method and precipitation dataset are appended by get_events_filename(),
# which the impact scripts use to read the file back.
LABEL_RESULTING_FILE = 'default_' + TARGET_TYPE
SAVE_AS_CSV = True

if DATASET == 'mobiliar':
    EXPOSURE_CATEGORIES = ['external']
    CLAIM_CATEGORIES = ['external', 'pluvial']
    CONFIG.set('DIR_EXPOSURE', CONFIG.get('DIR_EXPOSURE_MOBILIAR'))
    CONFIG.set('DIR_CLAIMS', CONFIG.get('DIR_CLAIMS_MOBILIAR'))
    CONFIG.set('YEAR_START', CONFIG.get('YEAR_START_MOBILIAR'))
    CONFIG.set('YEAR_END', CONFIG.get('YEAR_END_MOBILIAR'))
elif DATASET == 'gvz':
    EXPOSURE_CATEGORIES = ['all_buildings']
    CLAIM_CATEGORIES = ['likely_pluvial']
    CONFIG.set('DIR_EXPOSURE', CONFIG.get('DIR_EXPOSURE_GVZ'))
    CONFIG.set('DIR_CLAIMS', CONFIG.get('DIR_CLAIMS_GVZ'))
    CONFIG.set('YEAR_START', CONFIG.get('YEAR_START_GVZ'))
    CONFIG.set('YEAR_END', CONFIG.get('YEAR_END_GVZ'))
else:
    raise ValueError(f"Unknown damage dataset: {DATASET}")


def main():
    setup_logging(script_name='compute_claims_events_link')
    # Compute the claims and events link
    damages, events_to_remove = get_damages_linked_to_events()

    # Check that the damage categories are the same
    if not damages.claim_categories_are_for_type(CLAIM_CATEGORIES):
        logger.error("Error: the claim categories are not the same as the ones used for the "
              "events extraction.")
        return
    if not damages.exposure_categories_are_for_type(EXPOSURE_CATEGORIES):
        logger.error("Error: the exposure categories are not the same as the ones used for the "
              "events extraction.")
        return

    # Set the target variable value (occurrence or ratio)
    damages.set_target_variable_value(mode=TARGET_TYPE)

    # Assign the target value to the events
    events = Events()
    events.load_events_and_select_those_with_contracts(EVENTS_PATH, damages, EVENTS_TAG)
    events.check_precip_dataset(PRECIP_DATASET)
    events.set_target_values_from_damages(damages)
    events.set_contracts_number(damages)
    if events_to_remove is not None:
        events.remove_events(events_to_remove)
    else:
        logger.warning("No events to remove because the "
              "damages where loaded from pickle files.")
    events.remove_events_without_contracts()
    if METHOD == 'simple':
        events.remove_duplicates()

    nb_events = len(events.events)
    logger.info("Final number of events: %s", nb_events)

    # Save the events with target values to a pickle file
    filename = get_events_filename(DATASET, LABEL_RESULTING_FILE, METHOD,
                                   PRECIP_DATASET, extension='')
    events.save_to_pickle(filename=filename + '.pickle')
    if SAVE_AS_CSV:
        events.save_to_csv(filename=filename + '.csv')

    logger.info("Linked performed and saved to %s.", CONFIG.get('PICKLES_DIR'))


def get_damages_linked_to_events():
    year_start = CONFIG.get('YEAR_START')
    year_end = CONFIG.get('YEAR_END')
    label = LABEL_DAMAGE_LINK.replace(" ", "_")
    label = label + '_' + METHOD + PRECIP_SUFFIX
    filename = f'damages_{DATASET}_linked_{label}.pickle'
    file_path = Path(PICKLES_DIR + '/' + filename)

    if file_path.exists():
        logger.info("Link for %s already computed.", CRITERIA)
        if DATASET == 'mobiliar':
            damages = DamagesMobiliar(
                pickle_file=filename,
                year_start=year_start,
                year_end=year_end
            )
        elif DATASET == 'gvz':
            damages = DamagesGvz(
                pickle_file=filename,
                year_start=year_start,
                year_end=year_end
            )
        else:
            raise ValueError(f"Unknown damage dataset: {DATASET}")
        # Reload the list of events to remove that was saved alongside the
        # linked damages (see below). Without it, the cached path would skip the
        # removal and produce a different (polluted) events file than a fresh run.
        events_to_remove = _load_events_to_remove(filename)
        return damages, events_to_remove

    logger.info("Linking claims and events using method '%s'...", METHOD)
    if METHOD == 'classic':
        logger.info("Computing link for %s with window days %s...", CRITERIA, WINDOW_DAYS)

    if DATASET == 'mobiliar':
        damages = DamagesMobiliar(
            dir_exposure=CONFIG.get('DIR_EXPOSURE'),
            dir_claims=CONFIG.get('DIR_CLAIMS'),
            year_start=year_start,
            year_end=year_end
        )
    elif DATASET == 'gvz':
        damages = DamagesGvz(
            dir_exposure=CONFIG.get('DIR_EXPOSURE'),
            dir_claims=CONFIG.get('DIR_CLAIMS'),
            year_start=year_start,
            year_end=year_end
        )
    else:
        raise ValueError(f"Unknown damage dataset: {DATASET}")

    removed_claims = damages.select_categories_type(EXPOSURE_CATEGORIES, CLAIM_CATEGORIES)

    events = Events()
    events.load_events_and_select_those_with_contracts(EVENTS_PATH, damages, EVENTS_TAG)
    events.check_precip_dataset(PRECIP_DATASET)

    events_to_remove = damages.link_with_events(
        events,
        method=METHOD,
        criteria=CRITERIA,
        filename=filename,
        window_days=WINDOW_DAYS
    )

    events_removed_claims = events.get_events_for_removed_claims(removed_claims, damages)
    events_to_remove.extend(events_removed_claims)
    events_to_remove = list(set(events_to_remove))
    logger.info("Total number of events to remove: %s", len(events_to_remove))

    # Persist the list next to the linked damages pickle so that a cached re-run
    # applies the exact same removal (the damages pickle alone does not carry it).
    _save_events_to_remove(filename, events_to_remove)

    return damages, events_to_remove


def _events_to_remove_path(damages_filename):
    """Path of the sidecar file holding the events-to-remove list, derived from
    the linked-damages pickle name."""
    sidecar = damages_filename.replace('damages_', 'events_to_remove_', 1)
    return Path(PICKLES_DIR + '/' + sidecar)


def _save_events_to_remove(damages_filename, events_to_remove):
    path = _events_to_remove_path(damages_filename)
    with open(path, 'wb') as f:
        pickle.dump(events_to_remove, f)
    logger.info("Events to remove saved to %s", path)


def _load_events_to_remove(damages_filename):
    """Load the events-to-remove list saved alongside the linked damages. Returns
    None for legacy linkages saved before this list was persisted (the caller
    then warns that the removal cannot be reapplied)."""
    path = _events_to_remove_path(damages_filename)
    if not path.exists():
        logger.warning("No events-to-remove sidecar found at %s (legacy linkage). "
                       "Delete the linked damages pickle to recompute the link.", path)
        return None
    with open(path, 'rb') as f:
        events_to_remove = pickle.load(f)
    logger.info("Events to remove reloaded from %s (%s events)",
                path, len(events_to_remove))
    return events_to_remove


if __name__ == '__main__':
    main()
