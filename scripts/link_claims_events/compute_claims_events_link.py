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

from swafi.config import Config
from swafi.damages import Damages
from swafi.damages_mobiliar import DamagesMobiliar
from swafi.damages_gvz import DamagesGvz
from swafi.events import Events
from pathlib import Path

CONFIG = Config()

CRITERIA = ['prior', 'i_mean', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt']
LABEL_DAMAGE_LINK = 'original_w_prior_pluvial'
WINDOW_DAYS = [5, 3, 1]
PICKLES_DIR = CONFIG.get('PICKLES_DIR')
EVENTS_PATH = CONFIG.get('EVENTS_PATH')
TARGET_TYPE = 'occurrence'  # 'occurrence' or 'damage_ratio'
LABEL_RESULTING_FILE = 'original_w_prior_pluvial_' + TARGET_TYPE
SAVE_AS_CSV = False

DAMAGE_DATASET = 'gvz'  # 'mobiliar' or 'gvz'

if DAMAGE_DATASET == 'mobiliar':
    DAMAGE_CATEGORIES = ['external', 'pluvial']
    CONFIG.set('DIR_EXPOSURE', CONFIG.get('DIR_EXPOSURE_MOBILIAR'))
    CONFIG.set('DIR_CLAIMS', CONFIG.get('DIR_CLAIMS_MOBILIAR'))
    CONFIG.set('YEAR_START', CONFIG.get('YEAR_START_MOBILIAR'))
    CONFIG.set('YEAR_END', CONFIG.get('YEAR_END_MOBILIAR'))
elif DAMAGE_DATASET == 'gvz':
    DAMAGE_CATEGORIES = ['likely_pluvial']
    CONFIG.set('DIR_EXPOSURE', CONFIG.get('DIR_EXPOSURE_GVZ'))
    CONFIG.set('DIR_CLAIMS', CONFIG.get('DIR_CLAIMS_GVZ'))
    CONFIG.set('YEAR_START', CONFIG.get('YEAR_START_GVZ'))
    CONFIG.set('YEAR_END', CONFIG.get('YEAR_END_GVZ'))
else:
    raise ValueError(f"Unknown damage dataset: {DAMAGE_DATASET}")


def main():
    # Compute the claims and events link
    damages = get_damages_linked_to_events()

    # Check that the damage categories are the same
    assert damages.categories_are_for_type(DAMAGE_CATEGORIES)

    # Set the target variable value (occurrence or ratio)
    damages.set_target_variable_value(mode=TARGET_TYPE)

    # Assign the target value to the events
    events = Events()
    events.load_events_and_select_those_with_contracts(EVENTS_PATH, damages)
    events.set_target_values_from_damages(damages)
    events.set_contracts_number(damages)

    # Save the events with target values to a pickle file
    filename = f'events_with_{DAMAGE_DATASET}_target_values_{LABEL_RESULTING_FILE}'
    events.save_to_pickle(filename=filename + '.pickle')
    if SAVE_AS_CSV:
        events.save_to_csv(filename=filename + '.csv')

    print(f"Linked performed and saved to {CONFIG.get('PICKLES_DIR')}.")


def get_damages_linked_to_events():
    label = LABEL_DAMAGE_LINK.replace(" ", "_")
    filename = f'damages_{DAMAGE_DATASET}_linked_{label}.pickle'
    file_path = Path(PICKLES_DIR + '/' + filename)

    if file_path.exists():
        print(f"Link for {CRITERIA} already computed.")
        if DAMAGE_DATASET == 'mobiliar':
            damages = DamagesMobiliar(pickle_file=filename,
                                      year_start=CONFIG.get('YEAR_START'),
                                      year_end=CONFIG.get('YEAR_END'))
        elif DAMAGE_DATASET == 'gvz':
            damages = DamagesGvz(pickle_file=filename,
                                 year_start=CONFIG.get('YEAR_START'),
                                 year_end=CONFIG.get('YEAR_END'))
        else:
            raise ValueError(f"Unknown damage dataset: {DAMAGE_DATASET}")
        return damages

    print(f"Computing link for {CRITERIA}")
    if DAMAGE_DATASET == 'mobiliar':
        damages = DamagesMobiliar(dir_exposure=CONFIG.get('DIR_EXPOSURE'),
                                  dir_claims=CONFIG.get('DIR_CLAIMS'),
                                  year_start=CONFIG.get('YEAR_START'),
                                  year_end=CONFIG.get('YEAR_END'))
    elif DAMAGE_DATASET == 'gvz':
        damages = DamagesGvz(dir_exposure=CONFIG.get('DIR_EXPOSURE'),
                             dir_claims=CONFIG.get('DIR_CLAIMS'),
                             year_start=CONFIG.get('YEAR_START'),
                             year_end=CONFIG.get('YEAR_END'))
    else:
        raise ValueError(f"Unknown damage dataset: {DAMAGE_DATASET}")

    damages.select_categories_type(DAMAGE_CATEGORIES)

    events = Events()
    events.load_events_and_select_those_with_contracts(EVENTS_PATH, damages)

    damages.link_with_events(events, criteria=CRITERIA, filename=filename,
                             window_days=WINDOW_DAYS)

    return damages


if __name__ == '__main__':
    main()
