import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta

import core.damages
import core.events
import utils.plotting
from utils.config import Config
from pathlib import Path

config = Config()

criteria_list = [
    ['i_mean'],
    ['i_max'],
    ['p_sum'],
    ['i_mean', 'i_max', 'p_sum'],
    ['i_mean', 'i_max', 'p_sum', 'r_ts_win'],
    ['i_mean', 'i_max', 'p_sum', 'r_ts_evt'],
    ['i_mean', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'],
    ['i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'],
    ['prior', 'i_max', 'p_sum'],
    ['prior', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'],
]

labels = [
    'i_mean',
    'i_max',
    'p_sum',
    '3 qty',
    '3 qty + r_ts_win',
    '3 qty + r_ts_evt',
    'original',
    'v2',
    'v3',
]

window_days = [5, 3, 1]
#window_days = [9, 7, 5, 3, 1]

tmp_dir = config.get('TMP_DIR')

# Compute the different matching
for i, criteria in enumerate(criteria_list):
    filename = f'damages_matched_conf_{i}.pickle'
    file_path = Path(tmp_dir + '/' + filename)

    if file_path.exists():
        print(f"Criteria {criteria} already assessed.")
        continue

    print(f"Assessing criteria {criteria}")
    damages = core.damages.Damages(cid_file=config.get('CID_PATH'),
                                   dir_contracts=config.get('DIR_CONTRACTS'),
                                   dir_claims=config.get('DIR_CLAIMS'))
    damages.select_all_categories()

    events = core.events.Events()
    events.load_events_and_select_locations(config.get('EVENTS_PATH'), damages)

    damages.match_with_events(events, criteria=criteria, filename=filename,
                              window_days=window_days)

# Load the first pickle file and do some common work
damages = core.damages.Damages(pickle_file=f'damages_matched_conf_{0}.pickle')
total = damages.claims.eid.astype(bool).sum()
events = core.events.Events()
events.load_events_and_select_locations(config.get('EVENTS_PATH'), damages)
del damages

# Compare the events assigned
diff_count = np.zeros((len(criteria_list), len(criteria_list)))
for i_ref, criteria_ref in enumerate(criteria_list):
    filename_ref = f'damages_matched_conf_{i_ref}.pickle'
    df_ref = core.damages.Damages(pickle_file=filename_ref)
    assert total == df_ref.claims.eid.astype(bool).sum()

    # Compute and plot the time difference between the event and the claim date
    df_ref.merge_with_events(events)
    df_ref.compute_days_to_event_start('dt_start')
    df_ref.compute_days_to_event_center('dt_center')

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
        filename_diff = f'damages_matched_conf_{i_diff}.pickle'
        df_comp = core.damages.Damages(pickle_file=filename_diff)
        assert len(df_ref.claims) == len(df_comp.claims)

        diffs = df_ref.claims.eid - df_comp.claims.eid
        diff_count[i_ref, i_diff] = diffs.astype(bool).sum()

utils.plotting.plot_heatmap_differences(
    diff_count, total, labels, dir_output=config.get('OUTPUT_DIR'),
    title="Differences in the event-damage attribution")
