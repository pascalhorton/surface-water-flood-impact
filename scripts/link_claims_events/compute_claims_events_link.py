from swafi.config import Config
from swafi.damages import Damages
from swafi.events import Events
from pathlib import Path

CONFIG = Config()

CRITERIA = ['prior', 'i_mean', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt']
LABEL_DAMAGE_LINK = 'original_w_prior_pluvial'
WINDOW_DAYS = [5, 3, 1]
DAMAGE_CATEGORIES = ['external', 'pluvial']
PICKLES_DIR = CONFIG.get('PICKLES_DIR')
EVENTS_PATH = CONFIG.get('EVENTS_PATH')
LABEL_RESULTING_FILE = 'original_w_prior_pluvial'
SAVE_AS_CSV = False


def main():
    # Compute the claims and events link
    damages = get_damages_linked_to_events()

    # Check that the damage categories are the same
    assert damages.categories_are_for_type(DAMAGE_CATEGORIES)

    # Set the target variable value
    damages.set_target_variable_value(mode='occurrence')

    # Assign the target value to the events
    events = Events()
    events.load_events_and_select_those_with_contracts(EVENTS_PATH, damages)
    events.set_target_values_from_damages(damages)

    # Save the events with target values to a pickle file
    filename = f'events_with_target_values_{LABEL_RESULTING_FILE}'
    events.save_to_pickle(filename=filename + '.pickle')
    if SAVE_AS_CSV:
        events.save_to_csv(filename=filename + '.csv')

    print(f"Linked performed and saved to {CONFIG.get('PICKLES_DIR')}.")


def get_damages_linked_to_events():
    label = LABEL_DAMAGE_LINK.replace(" ", "_")
    filename = f'damages_linked_{label}.pickle'
    file_path = Path(PICKLES_DIR + '/' + filename)

    if file_path.exists():
        print(f"Link for {CRITERIA} already computed.")
        damages = Damages(pickle_file=filename)
        return damages

    print(f"Computing link for {CRITERIA}")
    damages = Damages(dir_contracts=CONFIG.get('DIR_CONTRACTS'),
                      dir_claims=CONFIG.get('DIR_CLAIMS'))
    damages.select_categories_type(DAMAGE_CATEGORIES)

    events = Events()
    events.load_events_and_select_those_with_contracts(EVENTS_PATH, damages)

    damages.link_with_events(events, criteria=CRITERIA, filename=filename,
                             window_days=WINDOW_DAYS)

    return damages


if __name__ == '__main__':
    main()
