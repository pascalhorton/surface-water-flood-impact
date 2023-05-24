import numpy as np
import pandas as pd
import core.damages
import core.events
import core.precipitation
import utils.plotting
from utils.config import Config
from pathlib import Path

CONFIG = Config()

PARAMETERS = [  # [label, [criteria], [window_days]]
    # Original
    ['original', ['i_mean', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'], [5, 3, 1]],
    # For the original temporal window ([5, 3, 1])
    ['v1', ['i_max', 'p_sum'], [5, 3, 1]],
    ['v2', ['i_max', 'p_sum', 'r_ts_win'], [5, 3, 1]],
    ['v3', ['i_max', 'p_sum', 'r_ts_evt'], [5, 3, 1]],
    ['v4', ['i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'], [5, 3, 1]],
    ['v5', ['prior', 'i_max', 'p_sum'], [5, 3, 1]],
    ['v6', ['prior', 'i_mean', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'], [5, 3, 1]],
    # Original with intermediate temporal window ([5, 3, 2, 1])
    ['original 4win', ['i_mean', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'], [5, 3, 2, 1]],
    # For an intermediate temporal window ([5, 3, 2, 1])
    ['v1 4win', ['i_max', 'p_sum'], [5, 3, 2, 1]],
    ['v2 4win', ['i_max', 'p_sum', 'r_ts_win'], [5, 3, 2, 1]],
    ['v3 4win', ['i_max', 'p_sum', 'r_ts_evt'], [5, 3, 2, 1]],
    ['v4 4win', ['i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'], [5, 3, 2, 1]],
    ['v5 4win', ['prior', 'i_max', 'p_sum'], [5, 3, 2, 1]],
    ['v6 4win', ['prior', 'i_mean', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'], [5, 3, 2, 1]]
]

DAMAGE_CATEGORIES = ['external', 'pluvial']

TMP_DIR = CONFIG.get('TMP_DIR')

PLOT_HISTOGRAMS = False
PLOT_MATRIX = True
PLOT_ALL_TIME_SERIES = False
PLOT_TIME_SERIES_DISAGREEMENT = True


def main():
    # Compute the different matching
    compute_link_and_save_to_pickle()

    # Load the first pickle file and do some common work
    damages = core.damages.Damages(
        pickle_file=f'damages_linked_{PARAMETERS[0][0].replace(" ", "_")}.pickle')
    events = core.events.Events()
    events.load_events_and_select_locations_with_contracts(CONFIG.get('EVENTS_PATH'), damages)
    del damages

    precip = None
    if PLOT_TIME_SERIES_DISAGREEMENT or PLOT_ALL_TIME_SERIES:
        # Precipitation data
        precip = core.precipitation.Precipitation(CONFIG.get('DIR_PRECIP'))

    # Compare the events assigned
    diff_count = np.zeros((len(PARAMETERS), len(PARAMETERS)))
    total = []
    for i_ref, params_ref in enumerate(PARAMETERS):
        label_ref = params_ref[0].replace(" ", "_")
        filename_ref = f'damages_linked_{label_ref}.pickle'
        df_ref = core.damages.Damages(pickle_file=filename_ref)
        total.append(df_ref.claims.eid.astype(bool).sum())

        # Compute and plot the time difference between the event and the claim date
        df_ref.merge_with_events(events)
        df_ref.compute_days_to_event_start('dt_start')
        df_ref.compute_days_to_event_center('dt_center')

        if PLOT_HISTOGRAMS:
            plot_histograms_time_differences(df_ref, label_ref)

        if PLOT_ALL_TIME_SERIES:
            plot_time_series(df_ref, precip, params_ref)

        # Compute the differences in events attribution with other criteria
        for i_diff, params_diff in enumerate(PARAMETERS):
            label_diff = params_diff[0].replace(" ", "_")
            filename_diff = f'damages_linked_{label_diff}.pickle'
            df_comp = core.damages.Damages(pickle_file=filename_diff)
            if len(df_ref.claims) >= len(df_comp.claims):
                df_merged_claims = pd.merge(df_ref.claims, df_comp.claims,
                                            how="left", on=['date_claim', 'cid'])
            else:
                df_merged_claims = pd.merge(df_ref.claims, df_comp.claims,
                                            how="right", on=['date_claim', 'cid'])

            diffs = df_merged_claims.eid_x - df_merged_claims.eid_y
            diff_count[i_ref, i_diff] = diffs.astype(bool).sum()

            if PLOT_TIME_SERIES_DISAGREEMENT:
                plot_time_series_different_events(
                    df_merged_claims, df_comp, df_ref, diffs, events, precip,
                    label_ref, label_diff)

    if PLOT_MATRIX:
        labels = [p[0] for p in PARAMETERS]
        utils.plotting.plot_heatmap_differences(
            diff_count, total, labels, dir_output=CONFIG.get('OUTPUT_DIR'),
            title="Differences in the event-damage attribution", fontsize=6)


def compute_link_and_save_to_pickle():
    for params in PARAMETERS:
        label = params[0].replace(" ", "_")
        criteria = params[1]
        window_days = params[2]
        filename = f'damages_linked_{label}.pickle'
        file_path = Path(TMP_DIR + '/' + filename)

        if file_path.exists():
            print(f"Criteria {criteria} already assessed.")
            continue

        print(f"Assessing criteria {criteria}")
        damages = core.damages.Damages(dir_contracts=CONFIG.get('DIR_CONTRACTS'),
                                       dir_claims=CONFIG.get('DIR_CLAIMS'))
        damages.select_categories_type(DAMAGE_CATEGORIES)

        events = core.events.Events()
        events.load_events_and_select_locations_with_contracts(CONFIG.get('EVENTS_PATH'), damages)

        damages.link_with_events(events, criteria=criteria, filename=filename,
                                 window_days=window_days)


def plot_time_series_different_events(df_merged_claims, df_comp, df_ref, diffs, events,
                                      precip, label_ref, label_diff):
    idx_diffs = diffs.to_numpy().nonzero()[0]
    df_comp.merge_with_events(events)
    for idx in idx_diffs:
        date_claim = df_merged_claims.iloc[idx]['date_claim']
        cid = df_merged_claims.iloc[idx]['cid']
        claim_1 = df_ref.claims.loc[(df_ref.claims.cid == cid) &
                                    (df_ref.claims.date_claim == date_claim)]
        claim_2 = df_comp.claims.loc[(df_comp.claims.cid == cid) &
                                     (df_comp.claims.date_claim == date_claim)]
        dir_output = CONFIG.get(
            'OUTPUT_DIR') + f'/Timeseries {label_ref} vs {label_diff}'
        utils.plotting.plot_claim_events_timeseries(
            [5, 3, 1], precip, claim_1, label_ref, claim_2,
            label_diff, dir_output=dir_output)


def plot_time_series(df_ref, precip, params):
    for idx in range(len(df_ref.claims)):
        claim = df_ref.claims.iloc[idx]
        label = params[0]
        dir_output = CONFIG.get('OUTPUT_DIR') + f'/Single timeseries {label}'
        window_days = [p[2] for p in PARAMETERS]
        utils.plotting.plot_claim_events_timeseries(
            window_days, precip, claim, label, dir_output=dir_output)


def plot_histograms_time_differences(df, label):
    dir_output = CONFIG.get('OUTPUT_DIR')
    title_start = f"Difference (in days) between the claim date and the " \
                  f"event start \n when using '{label}'"
    title_center = f"Difference (in days) between the claim date and the " \
                   f"event center \n when using '{label}'"

    if utils.plotting.output_file_exists(dir_output, title_start):
        return

    utils.plotting.plot_histogram_time_difference(
        df.claims, field_name='dt_start', dir_output=dir_output, title=title_start)
    utils.plotting.plot_histogram_time_difference(
        df.claims, field_name='dt_center', dir_output=dir_output, title=title_center)


if __name__ == '__main__':
    main()
