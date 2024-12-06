"""
This script is used to analyze the results of the link between claims and events.
"""

from swafi.config import Config
from swafi.damages_mobiliar import DamagesMobiliar
from swafi.damages_gvz import DamagesGvz
from swafi.events import Events
from swafi.precip_combiprecip import CombiPrecip
from swafi.utils.plotting import *
from pathlib import Path

CONFIG = Config()

PARAMETERS = [  # [label, [criteria], [window_days]]
    ['v1', ['i_max'], [5, 3, 1]],
    ['v2', ['p_sum'], [5, 3, 1]],
    ['v3', ['i_max', 'p_sum'], [5, 3, 1]],
    ['v4', ['i_max', 'p_sum', 'i_mean'], [5, 3, 1]],
    ['v5', ['i_max', 'p_sum', 'r_ts_evt'], [5, 3, 1]],
    ['v6', ['i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'], [5, 3, 1]],
    ['v7', ['i_mean', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'], [5, 3, 1]],
    ['v8', ['prior', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'], [5, 3, 1]],
    ['v9', ['prior', 'i_mean', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt'], [5, 3, 1]],
]

DATASET = 'gvz'

if DATASET == 'mobiliar':
    EXPOSURE_CATEGORIES = ['external']
    CLAIM_CATEGORIES = ['external', 'pluvial']
    CONFIG.set('YEAR_START', CONFIG.get('YEAR_START_MOBILIAR'))
    CONFIG.set('YEAR_END', CONFIG.get('YEAR_END_MOBILIAR'))
elif DATASET == 'gvz':
    EXPOSURE_CATEGORIES = ['all_buildings']
    CLAIM_CATEGORIES = ['likely_pluvial']
    CONFIG.set('YEAR_START', CONFIG.get('YEAR_START_GVZ'))
    CONFIG.set('YEAR_END', CONFIG.get('YEAR_END_GVZ'))
else:
    raise ValueError(f"Unknown damage dataset: {DATASET}")

PICKLES_DIR = CONFIG.get('PICKLES_DIR')

PLOT_HISTOGRAMS = False
PLOT_MATRIX = True
PLOT_ALL_TIME_SERIES = False
PLOT_TIME_SERIES_DISAGREEMENT = False


def main():
    # Compute the different matching
    compute_link_and_save_to_pickle()

    # Load the first pickle file and do some common work
    filename = f'damages_{DATASET}_linked_{PARAMETERS[0][0].replace(" ", "_")}.pickle'
    if DATASET == 'mobiliar':
        damages = DamagesMobiliar(pickle_file=filename,
                                  year_start=CONFIG.get('YEAR_START'),
                                  year_end=CONFIG.get('YEAR_END'))
    elif DATASET == 'gvz':
        damages = DamagesGvz(pickle_file=filename,
                             year_start=CONFIG.get('YEAR_START'),
                             year_end=CONFIG.get('YEAR_END'))
    else:
        raise ValueError(f"Unknown damage dataset: {DATASET}")

    events = Events()
    events.load_events_and_select_those_with_contracts(
        CONFIG.get('EVENTS_PATH'), damages, DATASET)
    del damages

    precip = None
    if PLOT_TIME_SERIES_DISAGREEMENT or PLOT_ALL_TIME_SERIES:
        # Precipitation data
        precip = CombiPrecip(CONFIG.get('YEAR_START'), CONFIG.get('YEAR_END'))

    # Compare the events assigned
    diff_count = np.zeros((len(PARAMETERS), len(PARAMETERS)))
    total = []
    for i_ref, params_ref in enumerate(PARAMETERS):
        label_ref = params_ref[0].replace(" ", "_")
        filename_ref = f'damages_{DATASET}_linked_{label_ref}.pickle'
        if DATASET == 'mobiliar':
            df_ref = DamagesMobiliar(pickle_file=filename_ref,
                                     year_start=CONFIG.get('YEAR_START'),
                                     year_end=CONFIG.get('YEAR_END'))
        elif DATASET == 'gvz':
            df_ref = DamagesGvz(pickle_file=filename_ref,
                                year_start=CONFIG.get('YEAR_START'),
                                year_end=CONFIG.get('YEAR_END'))
        else:
            raise ValueError(f"Unknown damage dataset: {DATASET}")

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
            filename_diff = f'damages_{DATASET}_linked_{label_diff}.pickle'
            if DATASET == 'mobiliar':
                df_comp = DamagesMobiliar(pickle_file=filename_diff,
                                          year_start=CONFIG.get('YEAR_START'),
                                          year_end=CONFIG.get('YEAR_END'))
            elif DATASET == 'gvz':
                df_comp = DamagesGvz(pickle_file=filename_diff,
                                     year_start=CONFIG.get('YEAR_START'),
                                     year_end=CONFIG.get('YEAR_END'))
            else:
                raise ValueError(f"Unknown damage dataset: {DATASET}")

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
        plot_heatmap_differences(
            diff_count, total, labels, dir_output=CONFIG.get('OUTPUT_DIR'),
            title="Differences in the event-damage attribution", fontsize=6)


def compute_link_and_save_to_pickle():
    for params in PARAMETERS:
        label = params[0].replace(" ", "_")
        criteria = params[1]
        window_days = params[2]
        filename = f'damages_{DATASET}_linked_{label}.pickle'
        file_path = Path(PICKLES_DIR + '/' + filename)

        if file_path.exists():
            print(f"Criteria {criteria} already assessed.")
            continue

        print(f"Assessing criteria {criteria}")
        if DATASET == 'mobiliar':
            damages = DamagesMobiliar(dir_exposure=CONFIG.get('DIR_EXPOSURE_MOBILIAR'),
                                      dir_claims=CONFIG.get('DIR_CLAIMS_MOBILIAR'),
                                      year_start=CONFIG.get('YEAR_START'),
                                      year_end=CONFIG.get('YEAR_END'))
        elif DATASET == 'gvz':
            damages = DamagesGvz(dir_exposure=CONFIG.get('DIR_EXPOSURE_GVZ'),
                                 dir_claims=CONFIG.get('DIR_CLAIMS_GVZ'),
                                 year_start=CONFIG.get('YEAR_START'),
                                 year_end=CONFIG.get('YEAR_END'))
        else:
            raise ValueError(f"Unknown damage dataset: {DATASET}")

        damages.select_categories_type(EXPOSURE_CATEGORIES, CLAIM_CATEGORIES)

        events = Events()
        events.load_events_and_select_those_with_contracts(
            CONFIG.get('EVENTS_PATH'), damages, DATASET)

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
        plot_claim_events_timeseries(
            [5, 3, 1], precip, claim_1, label_ref, claim_2,
            label_diff, dir_output=dir_output)


def plot_time_series(df_ref, precip, params):
    for idx in range(len(df_ref.claims)):
        claim = df_ref.claims.iloc[idx]
        label = params[0]
        dir_output = CONFIG.get('OUTPUT_DIR') + f'/Single timeseries {label}'
        window_days = [p[2] for p in PARAMETERS]
        plot_claim_events_timeseries(
            window_days, precip, claim, label, dir_output=dir_output)


def plot_histograms_time_differences(df, label):
    dir_output = CONFIG.get('OUTPUT_DIR')
    title_start = f"Difference (in days) between the claim date and the " \
                  f"event start \n when using '{label}'"
    title_center = f"Difference (in days) between the claim date and the " \
                   f"event center \n when using '{label}'"

    if output_file_exists(dir_output, title_start):
        return

    plot_histogram_time_difference(
        df.claims, field_name='dt_start', dir_output=dir_output, title=title_start)
    plot_histogram_time_difference(
        df.claims, field_name='dt_center', dir_output=dir_output, title=title_center)


if __name__ == '__main__':
    main()
