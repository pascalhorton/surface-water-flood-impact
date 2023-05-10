import numpy as np
import core.damages
import core.events
import utils.plotting
from utils.config import Config
from pathlib import Path

config = Config()

criteria_list = [
    ['i_max', 'p_sum'],
    ['i_max', 'p_sum'],
    ['i_max', 'p_sum', 'r_ts_win'],
    ['i_max', 'p_sum', 'r_ts_win'],
    ['i_max', 'p_sum', 'r_ts_evt'],
    ['i_max', 'p_sum', 'r_ts_evt'],
    ['i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'],
    ['i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'],
    ['prior', 'i_max', 'p_sum'],
    ['prior', 'i_max', 'p_sum'],
    ['prior', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'],
    ['prior', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'],
]

labels = [
    'v1',
    'v1 even',
    'v2',
    'v2 even',
    'v3',
    'v3 even',
    'v4',
    'v4 even',
    'v5',
    'v5 even',
    'v6',
    'v6 even',
]

window_days = [
    [5, 3, 1],
    [6, 4, 2],
    [5, 3, 1],
    [6, 4, 2],
    [5, 3, 1],
    [6, 4, 2],
    [5, 3, 1],
    [6, 4, 2],
    [5, 3, 1],
    [6, 4, 2],
]

damage_categories = ['external']

tmp_dir = config.get('TMP_DIR')

plot_histograms = True
plot_matrix = True

# Compute the different matching
for i, criteria in enumerate(criteria_list):
    label = labels[i].replace(" ", "_")
    filename = f'damages_linked_{label}.pickle'
    file_path = Path(tmp_dir + '/' + filename)

    if file_path.exists():
        print(f"Criteria {criteria} already assessed.")
        continue

    print(f"Assessing criteria {criteria}")
    damages = core.damages.Damages(dir_contracts=config.get('DIR_CONTRACTS'),
                                   dir_claims=config.get('DIR_CLAIMS'))
    damages.select_categories_type(damage_categories)

    events = core.events.Events()
    events.load_events_and_select_locations(config.get('EVENTS_PATH'), damages)

    damages.link_with_events(events, criteria=criteria, filename=filename,
                             window_days=window_days[i])

# Load the first pickle file and do some common work
damages = core.damages.Damages(
    pickle_file=f'damages_linked_{labels[0].replace(" ", "_")}.pickle')
total = damages.claims.eid.astype(bool).sum()
events = core.events.Events()
events.load_events_and_select_locations(config.get('EVENTS_PATH'), damages)
del damages

# Compare the events assigned
diff_count = np.zeros((len(criteria_list), len(criteria_list)))
for i_ref, criteria_ref in enumerate(criteria_list):
    label_ref = labels[i_ref].replace(" ", "_")
    filename_ref = f'damages_linked_{label_ref}.pickle'
    df_ref = core.damages.Damages(pickle_file=filename_ref)
    assert total == df_ref.claims.eid.astype(bool).sum()

    # Compute and plot the time difference between the event and the claim date
    df_ref.merge_with_events(events)
    df_ref.compute_days_to_event_start('dt_start')
    df_ref.compute_days_to_event_center('dt_center')

    if plot_histograms:
        utils.plotting.plot_histogram_time_difference(
            df_ref.claims, field_name='dt_start', dir_output=config.get('OUTPUT_DIR'),
            title=f"Difference (in days) between the claim date and the event start \n"
            f"when using '{labels[i_ref]}'")

        utils.plotting.plot_histogram_time_difference(
            df_ref.claims, field_name='dt_center', dir_output=config.get('OUTPUT_DIR'),
            title=f"Difference (in days) between the claim date and the event center \n"
            f"when using '{labels[i_ref]}'")

    # Compute the differences in events attribution with other criteria
    for i_diff, criteria_diff in enumerate(criteria_list):
        label_diff = labels[i_diff].replace(" ", "_")
        filename_diff = f'damages_linked_{label_diff}.pickle'
        df_comp = core.damages.Damages(pickle_file=filename_diff)
        assert len(df_ref.claims) == len(df_comp.claims)

        diffs = df_ref.claims.eid - df_comp.claims.eid
        diff_count[i_ref, i_diff] = diffs.astype(bool).sum()

if plot_matrix:
    utils.plotting.plot_heatmap_differences(
        diff_count, total, labels, dir_output=config.get('OUTPUT_DIR'),
        title="Differences in the event-damage attribution")
