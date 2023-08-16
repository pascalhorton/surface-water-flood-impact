import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd


def plot_random_forest_feature_importance(rf, features, importances, filename,
                                          dir_output=None, n_features=20):
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

    # Sort indices based on feature importance
    sorted_indices = np.argsort(importances)[::-1]

    # Select the top n features
    top_indices = sorted_indices[:n_features]
    top_importances = importances[top_indices]
    top_std = std[top_indices]
    top_feature_names = [features[idx] for idx in top_indices]

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(top_indices)), top_importances, tick_label=top_feature_names,
            yerr=top_std)
    plt.title(f'Top {n_features} Feature Importance - Mean Decrease in Impurity')
    plt.xlabel('Feature')
    plt.ylabel('Mean Decrease in Impurity')
    plt.xticks(rotation=90)
    plt.ylim([0, None])
    plt.tight_layout()

    _save_or_show(dir_output, filename)


def plot_heatmap_differences(data, total, labels, title, dir_output=None, fontsize=9):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data)

    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    threshold = im.norm(data.max()) / 2.
    kw = dict(ha='center', va='center', fontsize=fontsize)
    text_colors = ('white', 'black')

    for i in range(len(labels)):
        for j in range(len(labels)):
            kw.update(color=text_colors[int(im.norm(data[i, j]) > threshold)])
            total_val = total
            if isinstance(total, list):
                total_val = total[i]
            if i > j:
                pc = 100 * data[i, j] / total_val
                ax.text(j, i, f'{pc:.1f}%', **kw)
            elif j > i:
                ax.text(j, i, f'{int(data[i, j])}', **kw)

    ax.set_title(title)
    fig.tight_layout()

    _save_or_show(dir_output, title)


def plot_histogram_time_difference(claims, field_name, title, dir_output=None):
    hist_bins = claims[field_name].max() - claims[field_name].min() + 1
    hist_range = (claims[field_name].min() - 0.5, claims[field_name].max() + 0.5)
    counts, bins = np.histogram(claims[field_name], bins=hist_bins, range=hist_range)
    plt.stairs(counts, bins)
    plt.title(title)
    plt.xlabel("Difference in days")
    plt.ylabel("Count")
    plt.tight_layout()

    _save_or_show(dir_output, title)


def plot_claim_events_timeseries(window_days, precip, claim_1, label_1, claim_2=None,
                                 label_2=None, title=None, dir_output=None):
    cid = None
    claim_date = None

    if len(claim_1) > 0:
        if isinstance(claim_1, pd.DataFrame):
            claim_1 = claim_1.iloc[0]
        cid = claim_1.cid
        claim_date = claim_1.date_claim
    else:
        claim_1 = None

    if len(claim_2) > 0:
        if isinstance(claim_2, pd.DataFrame):
            claim_2 = claim_2.iloc[0]
        cid = claim_2.cid
        claim_date = claim_2.date_claim
    else:
        claim_2 = None

    # Get first event data
    e_1_dates = None
    precip_1 = None
    if claim_1 is not None:
        e_start_1 = claim_1.e_start - timedelta(hours=1)
        e_end_1 = claim_1.e_end + timedelta(hours=1)
        e_1_dur = (e_end_1 - e_start_1).total_seconds() / 3600
        e_1_dates = [e_start_1 + timedelta(hours=x) for x in range(int(e_1_dur) + 1)]
        precip_1 = precip.get_time_series(cid, e_start_1, e_end_1)

    # Get second event data
    e_2_dates = None
    precip_2 = None
    if claim_2 is not None:
        e_start_2 = claim_2.e_start - timedelta(hours=1)
        e_end_2 = claim_2.e_end + timedelta(hours=1)
        e_2_dur = (e_end_2 - e_start_2).total_seconds() / 3600
        e_2_dates = [e_start_2 + timedelta(hours=x) for x in range(int(e_2_dur) + 1)]
        precip_2 = precip.get_time_series(cid, e_start_2, e_end_2)

    # Get full window data
    delta_days_max = (max(window_days) - 1) / 2
    timedelta_max = timedelta(days=delta_days_max)
    win_start = datetime.combine(claim_date - timedelta_max, datetime.min.time())
    win_end = datetime.combine(claim_date + timedelta_max, datetime.max.time())
    precip_win = precip.get_time_series(cid, win_start, win_end, size=3)
    dates_win = [win_start + timedelta(hours=x) for x in range(len(precip_win))]

    # Plot precipitation
    fig, axs = plt.subplots(figsize=(12, 4))
    plt.plot(dates_win, precip_win, linewidth=1, color='0.1')
    if claim_1 is not None:
        plt.plot(e_1_dates, precip_1, label=label_1, linewidth=2)
    if claim_2 is not None:
        plt.plot(e_2_dates, precip_2, label=label_2, linewidth=2)

    # Add vertical span corresponding to the time windows
    for i_win, window in enumerate(window_days):
        delta_days = (window - 1) / 2
        win_start = datetime.combine(
            claim_date - timedelta(days=delta_days), datetime.min.time())
        win_end = datetime.combine(
            claim_date + timedelta(days=delta_days), datetime.max.time())
        alpha = 0.1 + 0.4 * i_win / len(window_days)
        axs.axvspan(win_start, win_end, facecolor='gray', alpha=alpha)

    plt.legend(loc='upper right')
    plt.ylabel("Precipitation [mm/h]")
    if title is not None:
        plt.title(title)
    plt.tight_layout()

    filename = f"{claim_date} {cid}"

    _save_or_show(dir_output, filename)


def output_file_exists(dir_output, title):
    if dir_output is None:
        return False
    dir_output = Path(dir_output)
    filename = re.sub(r'\W+', '', title.replace(' ', '_'))
    filepath_png = dir_output / (filename + '.png')
    filepath_pdf = dir_output / (filename + '.pdf')
    return filepath_png.exists() or filepath_pdf.exists()


def _save_or_show(dir_output, title):
    if dir_output is not None:
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)
        filename = re.sub(r'\W+', '', title.replace(' ', '_'))
        plt.savefig(dir_output / (filename + '.png'), dpi=600)
        plt.savefig(dir_output / (filename + '.pdf'), dpi=600)
        plt.close()
    else:
        plt.show()
